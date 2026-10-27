"""Teacher side of refiner v6c (report 124 §3; 2026-10-27 revision on corridor-cf-distill).

  T-D  CorridorHead(t=49, no agent attention) on the BEVFusion 64 m teacher BEV (kd_bev_0) -> occupancy logits
  T-M  CorridorHead(t=0) on the ReSMap front64 teacher BEV (kd_bev_1) -> drivable SDF
  T-E  RiskNet (object channels; no det tokens, + 3 path-wise GT area flags) on the BEVFusion BEV samples and the T-D / T-M
       fields along the path
Each is cross-fitted: copy f is trained on the tokens of fold f (cf_fold = crc32(log) % 2) and is used for the OTHER
fold, so every training token gets an out-of-fold (OOF) teacher.  Platt maps (per time bin) calibrate the occupancy
probabilities, the E_T logit channels and the T-M off-road probability; they are fitted on OOF outputs in stage T.

Baseline facts this relies on (report 125):
  * the dataloader already streams kd_bev_0 / kd_bev_1 [256, 100, 100] f16 on the student grid (row = forward 0..64 m,
    col = SSR x right) for the BEV distillation, so the teachers run ONLINE on them (no 59 GB field cache);
  * the teacher BEVs are z-scored per channel with the repo norm files (resources/km_corridor/norm_*_100x100.npz),
    the same statistics as the distillation; the teacher checkpoint records them.
  * there is no 64 m object pack in this branch: T-D runs without agent attention and T-E reads the teacher BEV
    samples instead of object rows (V14 open).
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from ..modules.corridor_head import CLIP, CorridorHead
from . import nets as N

logger = logging.getLogger(__name__)
TIME_BINS = (0, 10, 25, 41)                 # 0-1, 1-2.5, 2.5-4 s on the 41-step axis (report 124 D4 'Platt')
TTC_LAST_DT = 0.9                 # the TTC rule's largest projection (dt 0, 0.3, 0.6, 0.9 s; labels.py)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def load_norm(name: str) -> Dict[str, np.ndarray]:
    path = os.path.join(REPO, "resources", "km_corridor", f"norm_{name}_100x100.npz")
    d = np.load(path, allow_pickle=True)
    keys = list(d.files)
    mk = next(k for k in keys if "mean" in k.lower())
    sk = next(k for k in keys if "std" in k.lower())
    return {"mean": np.asarray(d[mk], np.float32).reshape(-1), "std": np.asarray(d[sk], np.float32).reshape(-1)}


class ZNorm(nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).clamp(min=1e-6).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] raw teacher BEV -> z-scored float32"""
        return (x.float() - self.mean) / self.std


class Platt(nn.Module):
    """p = sigmoid(a_k * x + b_k) per time bin k (x = a logit, or an SDF value for the T-M off-road map)"""

    def __init__(self, n: int = len(TIME_BINS) - 1, a: float = 1.0):
        super().__init__()
        self.register_buffer("a", torch.full((n,), float(a)))
        self.register_buffer("b", torch.zeros(n))

    def bins(self, steps: torch.Tensor) -> torch.Tensor:
        edges = torch.tensor(TIME_BINS[1:-1], device=steps.device)
        return torch.bucketize(steps, edges, right=True)

    def forward(self, x: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        """x [..., T] at 41-axis steps [T] -> probabilities"""
        k = self.bins(steps)
        return torch.sigmoid(self.a[k] * x + self.b[k])


class TeacherSet(nn.Module):
    """fold copies of T-D, T-M, T-E + norms + Platt maps.  Frozen in student training (eval mode, no grad)."""

    def __init__(self, cin: int = 256, bev_h: int = 100, bev_w: int = 100, folds: int = 2, hid_e: int = 192,
                 planes: int = 49):
        super().__init__()
        self.folds, self.bev_h, self.bev_w = folds, bev_h, bev_w
        self.td = nn.ModuleList([CorridorHead(cin, bev_h, bev_w, t=planes) for _ in range(folds)])
        self.tm = nn.ModuleList([CorridorHead(cin, bev_h, bev_w, t=0) for _ in range(folds)])
        self.te = nn.ModuleList([N.RiskNet(cin, hid_e, agent_dim=0, n_area=3) for _ in range(folds)])
        nb = load_norm("bevfusion")
        nm = load_norm("resmap")
        self.norm_det = ZNorm(nb["mean"], nb["std"])
        self.norm_map = ZNorm(nm["mean"], nm["std"])
        self.platt_occ = Platt(a=1.0)              # on T-D occupancy logits, by the plane's time bin
        self.platt_ovl = Platt(a=1.0)              # on T-E overlap logits
        self.platt_ttc = Platt(a=1.0)              # on T-E approach logits
        self.platt_off = Platt(a=-2.0)             # on the T-M min corner SDF (m): p(off-road) = sigmoid(a s + b), a < 0
        self.requires_grad_(False)

    def train(self, mode: bool = True) -> "TeacherSet":
        return super().train(False)                # always eval: BatchNorm running stats, no dropout

    # ------------------------------------------------------------------ #
    @staticmethod
    def oof_copy(cf_fold: torch.Tensor) -> torch.Tensor:
        """copy index used for each sample = the copy trained on the OTHER fold"""
        return (1 - cf_fold.long().clamp(0, 1))

    def _per_copy(self, mods: nn.ModuleList, copy: torch.Tensor, fn):
        outs = None
        for f in range(self.folds):
            sel = (copy == f).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            o = fn(mods[f], sel)
            if outs is None:
                outs = {k: v.new_zeros((copy.shape[0],) + v.shape[1:]) for k, v in o.items()}
            for k, v in o.items():
                outs[k][sel] = v
        return outs

    @torch.no_grad()
    def fields(self, kd_det: torch.Tensor, kd_map: torch.Tensor, copy: torch.Tensor) -> Dict[str, torch.Tensor]:
        """kd_det / kd_map [B, 256, 100, 100] raw (f16) teacher BEVs on the student grid; copy [B] ->
        occ_prob [B, 49, 200, 200] (Platt-calibrated), sdf_m [B, 200, 200] (T-M), tok_det [B, 10000, 256] z-scored"""
        zd = self.norm_det(kd_det)
        zm = self.norm_map(kd_map)
        tok_d = zd.flatten(2).transpose(1, 2)
        tok_m = zm.flatten(2).transpose(1, 2)
        occ = self._per_copy(self.td, copy, lambda m, s: {"o": m(tok_d[s])["corridor_occ"].float()})["o"]
        sdf = self._per_copy(self.tm, copy, lambda m, s: {"s": m(tok_m[s])["corridor_sdf"].float()})["s"]
        planes = torch.arange(1, occ.shape[1] + 1, device=occ.device)                   # plane j = step j + 1
        prob = self.platt_occ(occ.permute(0, 2, 3, 1), planes).permute(0, 3, 1, 2)
        return {"occ_prob": prob, "sdf_m": sdf, "tok_det": tok_d}

    @torch.no_grad()
    def path_outputs(self, S: torch.Tensor, flds: Dict[str, torch.Tensor], area: torch.Tensor, copy: torch.Tensor,
                     cin: int = 256) -> Dict[str, torch.Tensor]:
        """S [B, R, 41, 7] paths; area [B, R, T, 3] GT area flags along them ->
        p [B, R, T, 7] KD targets in E's units (probabilities on the 3 logit channels) and three gates [B, R, T]:
          g_obj  the overlap channel: all 4 corners of the box inside the teacher window 0 <= x <= 64, |y| <= 32
          g_ttc  the TTC-based channels (ttc, gap, vapp, px, py): g_obj AND the box shifted by speed * 0.9 s along the
                 heading (the TTC rule's farthest projection, labels.py) also inside the window (report 124 D2 /
                 report 127 §3.4: beyond 64 m the teacher sees nothing, so its 'no approach' there is not evidence)
          g_off  the off-road channel (= g_obj)"""
        reads = N.path_reads(S, flds["tok_det"], flds["sdf_m"], flds["occ_prob"], self.bev_h, self.bev_w, 64.0)
        E = self._per_copy(self.te, copy, lambda m, s: {"E": m({k: v[s] for k, v in reads.items()}, area=area[s])["E"]})["E"]
        t21 = N.T21.to(S.device)
        p_ovl = self.platt_ovl(E[..., 0], t21)
        p_ttc = self.platt_ttc(E[..., 1], t21)
        # T-M off-road probability from the min corner SDF (field_readout channel 0, already / CLIP in reads['ro'])
        p_off = self.platt_off(reads["ro"][..., 0] * CLIP, t21)
        p = torch.cat([p_ovl[..., None], p_ttc[..., None], E[..., 2:6], p_off[..., None]], -1)
        s21 = S[:, :, t21]
        corners = N.footprint5(s21)[..., 1:, :]                                         # [B, R, T, 4, 2]

        def win(c):
            return ((c[..., 0] >= 0) & (c[..., 0] <= 64.0) & (c[..., 1].abs() <= 32.0)).all(-1)

        inwin = win(corners)
        # the window is convex and the shift is linear in dt, so dt = 0 and the last dt (0.9 s) cover 0.3 / 0.6 s too
        shift = (torch.hypot(s21[..., 3], s21[..., 4]) * TTC_LAST_DT)[..., None]
        u = torch.stack([torch.cos(s21[..., 2]), torch.sin(s21[..., 2])], -1)
        inproj = inwin & win(corners + (u * shift)[..., None, :])
        return {"p": p, "g_obj": inwin.float(), "g_ttc": inproj.float(), "g_off": inwin.float()}


def load_teachers(path: str, cfg) -> TeacherSet:
    """stage-T checkpoint -> frozen TeacherSet (state_dict saved by tools/refiner_v6c/train_teachers.py)"""
    ck = torch.load(path, map_location="cpu")
    meta = ck.get("meta", {})
    ts = TeacherSet(int(cfg.embed_dims), int(cfg.bev_h), int(cfg.bev_w), folds=int(meta.get("folds", 2)),
                    hid_e=int(meta.get("hid_e", 192)), planes=int(meta.get("planes", 49)))
    # every stage-T phase saves the whole TeacherSet (untrained T-E included): only a checkpoint saved after the last
    # phase ('te': T-E trained + its Platt maps + the T-M off-road Platt) is a complete teacher (report 127 §3.2)
    need = ("fields", "platt_occ", "te")
    phases = meta.get("phases")
    if meta.get("phase_done") != "te" or (phases is not None and not all(x in phases for x in need)):
        raise RuntimeError(f"teacher checkpoint {path} is incomplete (phase_done={meta.get('phase_done')!r}, "
                           f"phases={phases}); use the checkpoint written after train_teachers.py --phase te "
                           f"(teachers_te.pt) of a run that went through {need}")
    if phases is None:
        logger.warning("teacher checkpoint %s has no phase record (written before 10-28): only phase_done='te' checked",
                       path)
    missing, unexpected = ts.load_state_dict(ck["state_dict"], strict=False)
    if unexpected or [k for k in missing if not k.startswith(("te.", "platt_"))]:
        raise RuntimeError(f"teacher checkpoint {path}: missing {missing[:8]}, unexpected {unexpected[:8]}")
    ts.loaded_parts = {"td": True, "tm": True, "te": not any(k.startswith("te.") for k in missing)}
    ts.requires_grad_(False)
    return ts.eval()
