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
    attn_feature  BEV cells weighted by where h_enc attends on the teacher
              BEV: sum_u a_T(u) d(A(F_S)(u), F_T(u))              (the method,
              spatial form: the reader still chooses *where*, but the BEV is
              matched cell by cell instead of through z)
    sens_feature  only the part of the gap that moves the reader's trajectory,
              measured on what the reader reads -- the layer-normed values
              v(u) = LN(W F(u) + b) of each cell:
              sum_u ||G_u (v_S(u) - v_T(u))||^2 / (2 sum_u ||G_u||_F^2),
              G_u = d traj / d v_T(u) at the teacher BEV (the method,
              planning-aware form: attn_feature still copies all C channels of
              an attended cell; this keeps only what changes the plan)

Why ``attn_feature``: z is a 1 x 256 summary trained only on the trajectory
loss.  Measured on navtest (report/22), a probe recovers the map raster from the
teacher BEV at road/walkway/centerline/crosswalk IoU 92/66/42/62 but from its
z at 76/37/23/28, which is what the student z already reaches -- the teacher's
map advantage does not survive z, so z distillation had nothing to transfer.
The reader's attention covers ~900 cells, 1.7x enriched on road, centerline
and crosswalk and follows the commanded side, so it is a planning-driven
selection that keeps the spatial layout.  The weights come from the frozen
reader on the teacher BEV (no grad) and are averaged over queries.

Why ``sens_feature``: attention says where the reader looks, not which part of
a cell's gap would change its trajectory.  With probes r_k ~ N(0, I) in
trajectory space (T x 3) and g_k = d(r_k . traj)/dv_T, E[(g_k(u) . x)^2] =
||G_u x||^2, so K probes estimate the per-cell first-order trajectory change
without the (T*3) x (H*W*d) Jacobian: K extra backward passes through the small
frozen reader per step.

Why the values and not the BEV: the reader layer-norms every cell before it
reads it, so it is (up to the in_proj bias) blind to a cell's scale, and a
Jacobian taken on the BEV has the teacher's own direction in its null space --
a student cell pointing the OPPOSITE way would cost nothing to first order.
Taking G at the normalised values and the gap exactly through W and LN keeps
the only linearisation inside attention and the decoder.  The student cell is
first rescaled to the teacher cell's norm (the student BEV is layer-normed,
|F_S(u)| ~15.7 everywhere, the teacher's ~8), which removes what is left of the
bias effect.  The denominator, 2 sum_u ||G_u||_F^2, is what an unrelated cell
would cost if the value gap were isotropic (two independent LN outputs differ by
2d in squared norm); real gaps are not, so an unrelated student scores O(1)
rather than exactly 1 (it only sets the scale, the balancer sets the share), and
a match in direction scores 0.  G is taken at the teacher BEV and
detached; the reader stays frozen.

Probes come from the distiller's own generator, so the training RNG stream
matches the other arms.  Validation runs in inference mode, where no Jacobian can
be formed; the term is reported as 0 there (the teacher cache has no validation
split anyway, teacher_valid is 0).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .readout import PlanningReadout, load_readout

KD_MODES = ("none", "readout", "random", "feature", "attn_feature", "sens_feature")
READER_MODES = ("readout", "attn_feature", "sens_feature")
KD_DISTANCES = ("cosine", "mse")
KD_CENTERS = ("none", "global", "command")
RANDOM_RANK = 16  # 16 x 16 = 256, the dimension of one readout query
NUM_COMMANDS = 4  # NAVSIM one-hot: left, straight, right, unknown


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
        self.center = str(getattr(config, "kd_center", "none"))
        if self.center not in KD_CENTERS:
            raise ValueError(f"kd_center must be one of {KD_CENTERS}, got {self.center!r}")
        if self.center != "none" and mode in ("feature", "attn_feature", "sens_feature"):
            # Centering is defined on the readout vector z; the feature control
            # matches the BEV itself, where there is no such vector.
            raise ValueError("kd_center is only defined for kd_mode readout|random (it acts on z)")
        self.center_momentum = float(getattr(config, "kd_center_momentum", 0.99))
        if not 0.0 <= self.center_momentum < 1.0:
            raise ValueError(f"kd_center_momentum must be in [0, 1), got {self.center_momentum}")
        self.sens_probes = int(getattr(config, "kd_sens_probes", 4))
        if mode == "sens_feature" and self.sens_probes < 1:
            raise ValueError(f"kd_sens_probes must be >= 1, got {self.sens_probes}")
        self._probe_gen: Optional[torch.Generator] = None
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
        if mode in READER_MODES:
            if not config.kd_readout_ckpt:
                raise ValueError(f"kd_mode={mode!r} needs kd_readout_ckpt (a Stage-1 readout)")
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

        if self.center != "none":
            # z is [B, Nq, d] for the readout and [B, 256] for the random control.
            shape = (reader.cfg.num_queries, reader.cfg.embed_dims) if mode == "readout" else (RANDOM_RANK ** 2,)
            rows = NUM_COMMANDS if self.center == "command" else 1
            for name in ("center_t", "center_s"):
                self.register_buffer(name, torch.zeros(rows, *shape))
            self.register_buffer("center_seen", torch.zeros(rows, dtype=torch.bool))

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

    def _center_rows(self, cmd: torch.Tensor) -> torch.Tensor:
        """Which running mean each sample belongs to."""
        if self.center == "command":
            return cmd.argmax(dim=1).clamp(max=NUM_COMMANDS - 1)
        return torch.zeros(cmd.shape[0], dtype=torch.long, device=cmd.device)

    def _update_center(self, buf: torch.Tensor, z: torch.Tensor, rows: torch.Tensor, valid: torch.Tensor) -> None:
        """EMA of z per row, from valid frames only; first sample adopted as is."""
        m = self.center_momentum
        for r in rows.unique():
            k = (rows == r) & (valid > 0)
            if not bool(k.any()):
                continue
            batch_mean = z[k].mean(0)
            if bool(self.center_seen[r]):
                buf[r].mul_(m).add_(batch_mean, alpha=1.0 - m)
            else:
                buf[r].copy_(batch_mean)
                self.center_seen[r] = True

    def _centered(self, z_s: torch.Tensor, z_t: torch.Tensor, cmd: torch.Tensor, valid: torch.Tensor):
        """Subtract each side's own running mean, so a constant per-command offset costs nothing."""
        if self.center == "none":
            return z_s, z_t
        rows = self._center_rows(cmd)
        for buf in (self.center_t, self.center_s):
            if buf.dtype != z_t.dtype or buf.device != z_t.device:
                buf.data = buf.data.to(device=z_t.device, dtype=z_t.dtype)
        if self.center_seen.device != z_t.device:
            self.center_seen.data = self.center_seen.data.to(z_t.device)
        # Center with the mean as it stood BEFORE this micro-batch: a mean that
        # already contains these very samples pulls each of them towards itself,
        # which at 4 frames per GPU would cancel most of the residual (and all of
        # it the first time a command is seen).
        with torch.no_grad():
            prev_t, prev_s, prev_seen = self.center_t.clone(), self.center_s.clone(), self.center_seen.clone()
            self._update_center(self.center_t, z_t, rows, valid)
            self._update_center(self.center_s, z_s.detach(), rows, valid)
            seen = prev_seen[rows].view(-1, *([1] * (z_t.dim() - 1)))
            mean_t = torch.where(seen, prev_t[rows], self.center_t[rows])
            mean_s = torch.where(seen, prev_s[rows], self.center_s[rows])
        return z_s - mean_s, z_t - mean_t

    @staticmethod
    def _values(reader: PlanningReadout, bev: torch.Tensor) -> torch.Tensor:
        """[L, B, H*W, d]: what each reader layer reads from every cell, LN(W F + b)."""
        tokens = reader.in_proj(bev.to(reader.in_proj.weight.dtype)).flatten(2).transpose(1, 2)
        return torch.stack([layer.memory_norm(tokens) for layer in reader.layers])

    def _sensitivity(
        self, reader: PlanningReadout, f_t: torch.Tensor, cmd: torch.Tensor, ego: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """[K, L, B, H*W, d] probe gradients d(r_k . traj)/dv at the teacher values (detached)."""
        if torch.is_inference_mode_enabled():
            return None
        if reader.cfg.ego_inject != "none" and ego is None:
            raise ValueError("kd_mode=sens_feature: this reader needs the ego status [B, 4]")
        if self._probe_gen is None or self._probe_gen.device != f_t.device:
            self._probe_gen = torch.Generator(device=f_t.device).manual_seed(0)
        leaves = []

        def as_leaf(_module, _inputs, out):
            leaf = out.detach().requires_grad_(True)
            leaves.append(leaf)
            return leaf

        handles = [layer.memory_norm.register_forward_hook(as_leaf) for layer in reader.layers]
        try:
            with torch.enable_grad():
                traj = reader(f_t.detach(), cmd, ego)["trajectory"].flatten(1)      # [B, T*3]
                probes = torch.randn(
                    self.sens_probes, traj.shape[1], device=traj.device, dtype=traj.dtype, generator=self._probe_gen
                )
                grads = [
                    torch.stack(torch.autograd.grad(
                        (traj * probes[k]).sum(), leaves, retain_graph=k < self.sens_probes - 1
                    ))
                    for k in range(self.sens_probes)
                ]
        finally:
            for h in handles:
                h.remove()
        return torch.stack(grads).detach()

    def _project(self, f: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B, 256] through the fixed random subspace."""
        x = torch.einsum("kc,bcn->bkn", self.rand_c.to(f.dtype), f.flatten(2))
        return torch.einsum("bkn,jn->bkj", x, self.rand_s.to(f.dtype)).flatten(1)

    def forward(
        self,
        bev_embed: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        iteration: int,
        ego: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.start_iter is None:
            self.start_iter = int(iteration)
        coef = self.weight * kd_ramp(iteration - self.start_iter, self.warmup, self.ramp)
        f_s = self.student_bev(bev_embed).float()
        f_t = targets["teacher_bev"].to(device=f_s.device, dtype=torch.float32)
        valid = targets["teacher_valid"].to(f_s.device).float().view(-1)
        cmd = targets["command"].to(f_s.device)

        plain = None
        if self.mode == "feature":
            d = self._dist(f_s, f_t)
        elif self.mode == "attn_feature":
            reader = self._reader_on(f_s)
            with torch.no_grad():
                _, attn = reader.encode(f_t, cmd, return_attn=True)   # [B, Nq, H*W]
                w = attn.float().mean(1)
                w = w / w.sum(1, keepdim=True).clamp(min=1e-12)
            if self.distance == "cosine":
                cell = 1.0 - F.cosine_similarity(f_s, f_t, dim=1)        # [B, H, W]
            else:
                cell = (f_s - f_t).pow(2).mean(1)
            d = (cell.flatten(1) * w).sum(1)
        elif self.mode == "sens_feature":
            reader = self._reader_on(f_s)
            jac = self._sensitivity(reader, f_t, cmd, ego)                        # [K, L, B, H*W, d]
            if jac is None:
                d = f_s.new_zeros(f_s.shape[0])
            else:
                t_norm = f_t.norm(dim=1, keepdim=True)
                s_scaled = f_s / f_s.norm(dim=1, keepdim=True).clamp(min=1e-6) * t_norm
                with torch.no_grad():
                    v_t = self._values(reader, f_t)
                    ref = 2.0 * jac.pow(2).sum(-1).sum(1).mean(0).sum(1)           # [B]
                dv = self._values(reader, s_scaled) - v_t                           # [L, B, H*W, d]
                change = (jac * dv.unsqueeze(0)).sum(-1).sum(1)                     # [K, B, H*W]
                d = change.pow(2).mean(0).sum(1) / ref.clamp(min=1e-12)
        elif self.mode == "random":
            z_s, z_t = self._project(f_s), self._project(f_t)
            plain = self._dist(z_s, z_t)
            z_s, z_t = self._centered(z_s, z_t, cmd, valid)
            d = self._dist(z_s, z_t) if self.center != "none" else plain
        else:
            reader = self._reader_on(f_s)
            z_s = reader.encode(f_s, cmd)                       # [B, Nq, d]
            with torch.no_grad():
                z_t = reader.encode(f_t, cmd)
            # cosine over the embedding axis: move it to dim 1
            plain = self._dist(z_s.transpose(1, 2), z_t.transpose(1, 2))
            z_s, z_t = self._centered(z_s, z_t, cmd, valid)
            d = self._dist(z_s.transpose(1, 2), z_t.transpose(1, 2)) if self.center != "none" else plain
        weighted = lambda x: (x * valid).sum() / valid.sum().clamp(min=1.0)
        raw = weighted(d)
        out = {
            "loss": raw * coef,
            "raw": raw.detach(),
            "coef": torch.tensor(coef, device=f_s.device),
            "valid_frac": valid.mean().detach(),
        }
        if self.center != "none" and plain is not None:
            # the distance a kd_center=none run would have minimised, for comparison
            out["raw_uncentered"] = weighted(plain).detach()
        return out
