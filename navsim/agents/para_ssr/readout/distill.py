"""Readout-space distillation term for PARA-SSR (report/19 s3, Stage 3).

    z_T = h_enc(F_T, cmd)            teacher cache, no grad
    z_S = h_enc(A(F_S), cmd)         same frozen h_enc, gradient flows into F_S
    L_distill = d(z_S, z_T)

``h_enc`` is frozen and deliberately NOT registered as a submodule of the agent:
it is not part of the student, must never be saved into or strictly loaded from
a student checkpoint, and must never be tied to the student's live planner (a
shared, trainable reader lets the planner absorb the mismatch and the loss
falls without F_S moving).  It follows the BEV's device on first use.

``kd_mode``
    readout   h_enc from a Stage-1 checkpoint                    (the method)
    random    fixed random 256-d projection of the BEV           (control:
              is it planning relevance or merely a 256-d bottleneck?)
    feature   no reader: masked MSE on the whole BEV             (control:
              full feature distillation, excess included)

The random control is a separable orthonormal projection (16 channel x 16
spatial directions = 256 dims), not an untrained reader: attention pooling with
random weights averages thousands of cells into an almost constant vector
(cosine distance ~1e-5 between unrelated BEVs), which would make the control
trivially weak rather than dimension-matched.

``A`` (``kd_adapter``) is an identity-initialised 1x1 conv, the capacity-limited
alignment adapter of design s05; it is a registered, trainable module of the
loss so it is checkpointed with the run.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .readout import PlanningReadout, load_readout

KD_MODES = ("none", "readout", "random", "feature")
KD_DISTANCES = ("cosine", "mse")
RANDOM_RANK = 16  # 16 x 16 = 256, the dimension of one readout query


def kd_ramp(iteration: int, warmup: int, ramp: int) -> float:
    """0 during ``warmup`` micro-batches, then linear to 1 over ``ramp``."""
    if iteration < warmup:
        return 0.0
    if ramp <= 0:
        return 1.0
    return min(1.0, (iteration - warmup + 1) / ramp)


class ReadoutDistiller(nn.Module):
    def __init__(self, config):
        super().__init__()
        mode = config.kd_mode
        if mode not in KD_MODES or mode == "none":
            raise ValueError(f"kd_mode must be one of {KD_MODES[1:]}, got {mode!r}")
        if config.kd_distance not in KD_DISTANCES:
            raise ValueError(f"kd_distance must be one of {KD_DISTANCES}, got {config.kd_distance!r}")
        self.mode = mode
        self.distance = config.kd_distance
        self.weight = float(config.kd_weight)
        self.warmup = int(config.kd_warmup_iters)
        self.ramp = int(config.kd_ramp_iters)
        self.bev_h, self.bev_w, self.dims = config.bev_h, config.bev_w, config.embed_dims
        # Loss iteration at the first distillation step.  The schedule counts
        # from here, not from 0: a fine-tune restores the base run's iteration
        # counter, which would otherwise skip the warm-up.  Checkpointed through
        # ParaSSRLoss's extra state so a resumed KD run does not restart it.
        self.start_iter: Optional[int] = None

        reader: Optional[PlanningReadout] = None
        if mode == "readout":
            if not config.kd_readout_ckpt:
                raise ValueError("kd_mode='readout' needs kd_readout_ckpt (a Stage-1 readout)")
            reader = load_readout(config.kd_readout_ckpt)
        elif mode == "random":
            g = torch.Generator().manual_seed(int(config.kd_random_seed))
            rc = torch.linalg.qr(torch.randn(self.dims, RANDOM_RANK, generator=g))[0].T
            rs = torch.linalg.qr(torch.randn(self.bev_h * self.bev_w, RANDOM_RANK, generator=g))[0].T
            self.register_buffer("rand_c", rc.contiguous(), persistent=False)   # [16, C]
            self.register_buffer("rand_s", rs.contiguous(), persistent=False)   # [16, H*W]
        if reader is not None:
            if (reader.cfg.bev_h, reader.cfg.bev_w) != (config.bev_h, config.bev_w):
                raise ValueError("readout grid does not match the student BEV grid")
            reader.eval().requires_grad_(False)
        self.__dict__["_reader"] = reader  # unregistered on purpose, see module doc

        self.adapter = None
        if config.kd_adapter:
            self.adapter = nn.Conv2d(self.dims, self.dims, kernel_size=1)
            with torch.no_grad():
                self.adapter.weight.copy_(torch.eye(self.dims).view(self.dims, self.dims, 1, 1))
                self.adapter.bias.zero_()

    # the reader is not a submodule, so .to()/.cuda() on the agent miss it
    def _reader_on(self, like: torch.Tensor) -> PlanningReadout:
        r = self.__dict__["_reader"]
        p = next(r.parameters())
        if p.device != like.device:
            r.to(like.device)
        return r

    def student_bev(self, bev_embed: torch.Tensor) -> torch.Tensor:
        """PARA-SSR ``bev_embed`` [B, H*W, C] -> [B, C, H, W] (forward, right)."""
        b = bev_embed.shape[0]
        f = bev_embed.view(b, self.bev_h, self.bev_w, self.dims).permute(0, 3, 1, 2)
        return self.adapter(f) if self.adapter is not None else f

    def _dist(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Per-sample distance; vectors along dim 1, averaged over the rest."""
        bs = a.shape[0]
        if self.distance == "cosine":
            return (1.0 - F.cosine_similarity(a, b, dim=1)).reshape(bs, -1).mean(1)
        return (a - b).pow(2).reshape(bs, -1).mean(1)

    def _project(self, f: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B, 256] through the fixed random subspace."""
        x = torch.einsum("kc,bcn->bkn", self.rand_c.to(f.dtype), f.flatten(2))
        return torch.einsum("bkn,jn->bkj", x, self.rand_s.to(f.dtype)).flatten(1)

    def forward(
        self, bev_embed: torch.Tensor, targets: Dict[str, torch.Tensor], iteration: int
    ) -> Dict[str, torch.Tensor]:
        if self.start_iter is None:
            self.start_iter = int(iteration)
        coef = self.weight * kd_ramp(iteration - self.start_iter, self.warmup, self.ramp)
        f_s = self.student_bev(bev_embed).float()
        f_t = targets["teacher_bev"].to(device=f_s.device, dtype=torch.float32)
        valid = targets["teacher_valid"].to(f_s.device).float().view(-1)
        cmd = targets["command"].to(f_s.device)

        if self.mode == "feature":
            d = self._dist(f_s, f_t)
        elif self.mode == "random":
            d = self._dist(self._project(f_s), self._project(f_t))
        else:
            reader = self._reader_on(f_s)
            z_s = reader.encode(f_s, cmd)                       # [B, Nq, d]
            with torch.no_grad():
                z_t = reader.encode(f_t, cmd)
            # cosine over the embedding axis: move it to dim 1
            d = self._dist(z_s.transpose(1, 2), z_t.transpose(1, 2))
        raw = (d * valid).sum() / valid.sum().clamp(min=1.0)
        return {
            "loss": raw * coef,
            "raw": raw.detach(),
            "coef": torch.tensor(coef, device=f_s.device),
            "valid_frac": valid.mean().detach(),
        }
