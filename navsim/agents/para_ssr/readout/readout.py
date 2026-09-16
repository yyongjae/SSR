"""The planning readout ``h = h_dec o h_enc``.

    h_enc(F, cmd) = z        reads a BEV, returns the planning summary z
    h_dec(z, ego) = tau      adds ego status, regresses the trajectory

``h`` is not a new planner; it is the smallest thing that can plan from a BEV,
used as a measuring instrument (report/19 s2).  One skeleton, capacity is the
only ablation axis:

    h0   attention pooling, no FFN, linear head          (the "linear probe")
    h1   one cross-attention + FFN, 2-layer MLP head      (default)
    h2   three cross-attention + FFN, 2-layer MLP head    (PARA-SSR planner size)

Design decisions that the rest of the pipeline relies on:

* **The BEV is a set of tokens with metric positions.**  The position of every
  cell is an MLP of its centre in metres, not a learned per-index table, so the
  same ``h`` reads any grid that covers the same ROI.  With ``num_queries=1``
  dense attention is O(cells).
* **Ego status enters after z** (``ego_inject="late"``).  z is then a function of
  the BEV and the command only, so ``||z_T - z_S||`` cannot be lowered by the
  ego status both sides share, and the readout cannot plan from ego alone while
  ignoring the BEV.  ``"early"`` reproduces PARA-SSR's placement for ablation.
* **The command enters the query** (``cmd_inject="query"``): it decides *where*
  to read, which is part of reading.  ``"late"`` is the ablation.
* ``use_bev=False`` is the ego floor ``h_ego``: identical everything, BEV unread.

Output follows PARA-SSR: per-step offsets in NAVSIM ego axes, expanded to the
four command branches so ``compute_plan_loss`` applies unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class ReadoutConfig:
    in_channels: int = 256
    embed_dims: int = 256
    num_heads: int = 8
    ffn_channels: int = 512
    num_layers: int = 1
    use_ffn: bool = True
    head_layers: int = 2
    num_queries: int = 1
    ego_inject: str = "late"      # late | early | none
    cmd_inject: str = "query"     # query | late
    use_bev: bool = True
    # student ROI / grid: (x_right_min, y_fwd_min, z_min, x_right_max, y_fwd_max, z_max)
    pc_range: Tuple[float, ...] = (-32.0, 0.0, -2.0, 32.0, 32.0, 2.0)
    bev_h: int = 50
    bev_w: int = 100
    fut_ts: int = 8
    traj_dims: int = 3
    num_cmd: int = 4
    ego_dims: int = 4

    def __post_init__(self):
        if self.ego_inject not in ("late", "early", "none"):
            raise ValueError(f"ego_inject must be late|early|none, got {self.ego_inject!r}")
        if self.cmd_inject not in ("query", "late"):
            raise ValueError(f"cmd_inject must be query|late, got {self.cmd_inject!r}")
        if self.head_layers not in (1, 2):
            raise ValueError(f"head_layers must be 1 or 2, got {self.head_layers}")
        self.pc_range = tuple(float(v) for v in self.pc_range)


READOUT_PRESETS: Dict[str, Dict] = {
    "h0": dict(num_layers=1, use_ffn=False, head_layers=1),
    "h1": dict(num_layers=1, use_ffn=True, head_layers=2),
    "h2": dict(num_layers=3, use_ffn=True, head_layers=2),
}


def metric_cell_centres(pc_range: Sequence[float], bev_h: int, bev_w: int) -> torch.Tensor:
    """``[bev_h * bev_w, 2]`` cell centres (x_right, y_forward) in metres.

    Row-major over (forward row, right column), matching PARA-SSR's flattened
    ``bev_embed`` and ``BevCache.bev``.
    """
    x0, y0, x1, y1 = pc_range[0], pc_range[1], pc_range[3], pc_range[4]
    xs = x0 + (torch.arange(bev_w, dtype=torch.float32) + 0.5) * (x1 - x0) / bev_w
    ys = y0 + (torch.arange(bev_h, dtype=torch.float32) + 0.5) * (y1 - y0) / bev_h
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2)


class ReadoutLayer(nn.Module):
    """Pre-LN: query <- cross-attention over BEV tokens, then optional FFN."""

    def __init__(self, dims: int, heads: int, ffn_channels: int, use_ffn: bool):
        super().__init__()
        self.query_norm = nn.LayerNorm(dims)
        self.memory_norm = nn.LayerNorm(dims)
        self.attn = nn.MultiheadAttention(dims, heads, batch_first=True)
        self.ffn = None
        if use_ffn:
            self.ffn_norm = nn.LayerNorm(dims)
            self.ffn = nn.Sequential(
                nn.Linear(dims, ffn_channels), nn.ReLU(inplace=True), nn.Linear(ffn_channels, dims)
            )

    def forward(self, q, q_pos, tokens, pos):
        value = self.memory_norm(tokens)
        upd, weights = self.attn(
            self.query_norm(q) + q_pos, value + pos, value,
            need_weights=True, average_attn_weights=True,
        )
        q = q + upd
        if self.ffn is not None:
            q = q + self.ffn(self.ffn_norm(q))
        return q, weights


class PlanningReadout(nn.Module):
    def __init__(self, cfg: ReadoutConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.embed_dims
        self.in_proj = nn.Conv2d(cfg.in_channels, d, kernel_size=1)
        self.register_buffer(
            "cell_xy", metric_cell_centres(cfg.pc_range, cfg.bev_h, cfg.bev_w), persistent=False
        )
        # metres -> roughly [-1, 1] before the MLP
        half = max(abs(v) for v in cfg.pc_range[:2] + cfg.pc_range[3:5])
        self.pos_scale = 1.0 / half
        self.pos_mlp = nn.Sequential(nn.Linear(2, d), nn.ReLU(inplace=True), nn.Linear(d, d))

        self.query = nn.Embedding(cfg.num_queries, d)
        self.query_pos = nn.Embedding(cfg.num_queries, d)
        self.cmd_embed = nn.Embedding(cfg.num_cmd, d)
        self.layers = nn.ModuleList(
            ReadoutLayer(d, cfg.num_heads, cfg.ffn_channels, cfg.use_ffn)
            for _ in range(cfg.num_layers)
        )
        self.ego_mlp = (
            nn.Sequential(nn.Linear(cfg.ego_dims, d), nn.ReLU(inplace=True), nn.Linear(d, d))
            if cfg.ego_inject != "none" else None
        )

        flat = cfg.num_queries * d
        out = cfg.fut_ts * cfg.traj_dims
        self.out_norm = nn.LayerNorm(d)
        if cfg.head_layers == 1:
            self.head = nn.Linear(flat, out)
        else:
            self.head = nn.Sequential(nn.Linear(flat, d), nn.ReLU(inplace=True), nn.Linear(d, out))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _cmd_index(cmd: torch.Tensor) -> torch.Tensor:
        return cmd.reshape(cmd.shape[0], -1).argmax(dim=-1)

    def _ego_term(self, ego: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
        if ego is None:
            raise ValueError("ego status [B, 4] is required unless ego_inject='none'")
        return self.ego_mlp(ego.to(like.dtype)).unsqueeze(1)

    def encode(
        self,
        bev: Optional[torch.Tensor],
        cmd: torch.Tensor,
        ego: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        """``bev`` [B, C, bev_h, bev_w] (student layout) -> z [B, num_queries, d].

        ``ego`` is consumed here only for ``ego_inject="early"``.
        """
        cfg = self.cfg
        bs = cmd.shape[0]
        dtype = self.query.weight.dtype
        q = self.query.weight.unsqueeze(0).expand(bs, -1, -1)
        if cfg.cmd_inject == "query":
            q = q + self.cmd_embed(self._cmd_index(cmd)).unsqueeze(1)
        if cfg.ego_inject == "early":
            q = q + self._ego_term(ego, q)
        attn = None
        if cfg.use_bev:
            if bev is None:
                raise ValueError("this readout reads a BEV; pass bev or build it with use_bev=False")
            if bev.shape[-2:] != (cfg.bev_h, cfg.bev_w):
                raise ValueError(
                    f"expected BEV grid {(cfg.bev_h, cfg.bev_w)} (forward, right), "
                    f"got {tuple(bev.shape[-2:])}"
                )
            tokens = self.in_proj(bev.to(dtype)).flatten(2).transpose(1, 2)
            pos = self.pos_mlp(self.cell_xy.to(dtype) * self.pos_scale).unsqueeze(0)
            q_pos = self.query_pos.weight.unsqueeze(0)
            for layer in self.layers:
                q, attn = layer(q, q_pos, tokens, pos)
        return (q, attn) if return_attn else q

    def decode(self, z: torch.Tensor, cmd: torch.Tensor, ego: Optional[torch.Tensor]) -> torch.Tensor:
        """z [B, Nq, d] -> per-step offsets [B, fut_ts, traj_dims]."""
        cfg = self.cfg
        u = z
        if cfg.ego_inject == "late":
            u = u + self._ego_term(ego, u)
        if cfg.cmd_inject == "late":
            u = u + self.cmd_embed(self._cmd_index(cmd)).unsqueeze(1)
        out = self.head(self.out_norm(u).flatten(1))
        return out.view(z.shape[0], cfg.fut_ts, cfg.traj_dims)

    def forward(self, bev, cmd, ego) -> Dict[str, torch.Tensor]:
        z, attn = self.encode(bev, cmd, ego, return_attn=True)
        offsets = self.decode(z, cmd, ego)
        bs = offsets.shape[0]
        return {
            "z": z,
            "attn": attn,
            # same contract as PARA-SSR's planner: one trajectory, every branch
            "ego_fut_preds": offsets.unsqueeze(1).expand(bs, self.cfg.num_cmd, -1, -1),
            "trajectory": offsets.cumsum(dim=1),
        }


# ---------------------------------------------------------------------- #
def build_readout(preset: str = "h1", **overrides) -> PlanningReadout:
    if preset not in READOUT_PRESETS:
        raise ValueError(f"unknown readout preset {preset!r}; choose from {sorted(READOUT_PRESETS)}")
    kwargs = dict(READOUT_PRESETS[preset])
    kwargs.update(overrides)
    return PlanningReadout(ReadoutConfig(**kwargs))


def readout_checkpoint(model: PlanningReadout, **extra) -> Dict:
    return {"readout_config": asdict(model.cfg), "state_dict": model.state_dict(), **extra}


def load_readout(path, map_location="cpu") -> PlanningReadout:
    ckpt = torch.load(path, map_location=map_location)
    model = PlanningReadout(ReadoutConfig(**ckpt["readout_config"]))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model
