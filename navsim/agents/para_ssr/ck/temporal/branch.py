"""refiner-v3 TemporalBranch: the shared temporal component of the CK refiner (docs/refiner_v3/SPEC.md section 4.5).

Data flow (per training step)
  scene(objs)   once per token set: causal object rows -> trim_objects -> Forecast (learned / CA / oracle), float32,
                autocast disabled.  -> TemporalCtx(fc = what the raster reads, fc_raw = learned forecast with grad
                for the forecast loss, n_obj).
  features(ctx, geom, flat_tau, T, Kc)
                per candidate set (cand, extra, rescore): rasterize + gap_scalars on the candidates' own corridor
                geometry (float32, autocast off, under no_grad unless cfg.fc_backprop), then the SHARED encoder
                st_enc (Conv 18-32-48-48, GroupNorm, GELU; caller's autocast) -> [B, 48 stations, 48 * 21] ->
                st_proj (Linear 1008 -> d, ZERO-init) and gap_proj (Linear 20 -> d, ZERO-init).
                -> st_add [B, 48, d] (added to the 48 station tokens), ud_add [B, d] (added to the draft token).
  hook(ctx, T, Kc) -> tfeat(geom, flat) for CKTrunk.candidates (the model.py integration point):
                x = cat([draft_mlp(ud) + ud_add, st + st_add]); zero-init projections make an untrained branch an
                exact no-op on the trunk.

Frozen learned forecaster (cfg.fc_frozen: learned + fc_train 'frozen', the Step-1 default since review fix D3)
  the forecaster's weights come from a pre-training stage (tools/ck/v3/pretrain_forecaster.py, loaded by train_ck2
  via cfg.fc_init) and stay fixed: scene() runs it under no_grad, train_ck2 sets requires_grad False, adds no
  forecast loss and logs the forecast stats only.  fc_train 'joint' keeps the old behaviour (forecast loss trains it).

Initialisation / arms
  shared modules (identical in arms a, b, d) under fork_rng + manual_seed(seed + TEMPORAL_SEED_OFFSET);
  arm a (forecaster 'learned') builds LearnedForecaster afterwards under seed + FORECASTER_SEED_OFFSET;
  arms b ('ca') and d ('oracle_gt') build NO forecaster and register the persistent buffer ca_b [2, 10]
  (vehicle, VRU CA Laplace scales; read from cfg.ca_scales at build time, so the checkpoint carries them).
  No global RNG is consumed by building the branch.

Input guards (scene)
  objs must hold OBJ_INPUT_KEYS; PRIV_INPUT_KEYS only for the oracle forecaster (required there); TARGET_KEYS never.
  Causality: a valid t0 row must have age 0 and a valid past row age > 0 (else ValueError: a future frame or a
  mislabelled frame reached the model).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .config import (COL, FC_STEPS, OBJ_INPUT_KEYS, PRIV_INPUT_KEYS, TARGET_KEYS, TEMPORAL_SEED_OFFSET, TemporalCfg)
from .forecast import Forecast, LearnedForecaster, ca_forecast, effective_mask, oracle_forecast, trim_objects
from .raster import C_R, N_GAP, gap_scalars, rasterize

Tensor = torch.Tensor
N_ST_TOK = 48
N_RT_TOK = 21
ST_CH = (32, 48, 48)
CA_SCALE_GROUPS = ("vehicle", "vru")


@dataclass
class TemporalCtx:
    fc: Forecast                  # read by raster / gap (detached unless cfg.fc_backprop)
    fc_raw: Optional[Forecast]    # with grad, for the forecast loss (learned only, else None)
    n_obj: int                    # trimmed t0 slot count of fc (>= 1 even with no object: the trim keeps one slot)
    n_valid: int = 0              # valid t0 slots over the batch (0 for an object-free batch)


def load_ca_b(path: Union[str, Path], source: Optional[str] = None) -> Tensor:
    """ca_scales.json (SPEC 3.6) -> float32 [2, 10] (vehicle, vru); every value finite and > 0.  The file records its
    object source and K_OBJ (review fix: source-keyed scales); ``source`` given -> a file of another source raises."""
    d = json.loads(Path(path).read_text())
    fsrc = d.get("source", "bevfusion")                       # files written before the field existed: bevfusion
    if source is not None and fsrc != source:
        raise ValueError(f"{path}: CA scales fitted for source {fsrc!r}, the branch uses {source!r}")
    b = np.asarray([d["b"][g] for g in CA_SCALE_GROUPS], np.float64)
    if b.shape != (2, FC_STEPS) or not np.isfinite(b).all() or (b <= 0).any():
        raise ValueError(f"{path}: bad CA scales {b.shape}")
    return torch.as_tensor(b, dtype=torch.float32)


def trim_forecast(fc: Forecast) -> Forecast:
    """Drop t0 slots after the last valid one in the batch (they only carry past-frame padding)."""
    any_slot = fc.valid.any(0)
    k = int(torch.nonzero(any_slot).max()) + 1 if bool(any_slot.any()) else 1
    if k == fc.mu.shape[1]:
        return fc
    kw = {}
    for name in ("mu", "b", "pi", "present", "c0", "v0", "h0", "L", "W", "group", "valid", "log_b", "logit_pi"):
        v = getattr(fc, name)
        kw[name] = None if v is None else v[:, :k]
    return replace(fc, **kw)


def objs_from_batch(batch: Dict, cfg: TemporalCfg, device=None) -> Optional[Dict[str, Tensor]]:
    """The ONLY keys of a collated batch that may reach the temporal branch: OBJ_INPUT_KEYS, plus PRIV_INPUT_KEYS for
    the oracle forecaster.  None when the branch is off."""
    if not cfg.on:
        return None
    keys = OBJ_INPUT_KEYS + (PRIV_INPUT_KEYS if cfg.forecaster == "oracle_gt" else ())
    missing = [k for k in keys if k not in batch]
    if missing:
        raise KeyError(f"batch lacks temporal input keys {missing}")
    out = {}
    for k in keys:
        v = batch[k]
        v = torch.as_tensor(v) if not isinstance(v, torch.Tensor) else v
        out[k] = v.to(device, non_blocking=True) if device is not None else v
    return out


def check_causal(obj_in: Tensor, obj_mask: Tensor, frame_valid: Tensor) -> None:
    """Valid t0 rows must have age 0, valid past rows age > 0 (strictly earlier than t0)."""
    m = effective_mask(obj_mask, frame_valid)
    age = obj_in[..., COL["age"]]
    bad0 = m[:, 0] & (age[:, 0] != 0)
    badp = m[:, 1:] & ~(age[:, 1:] > 0)
    if bool(bad0.any() | badp.any()):
        raise ValueError(f"non-causal object input: {int(bad0.sum())} t0 rows with age != 0, {int(badp.sum())} past "
                         f"rows with age <= 0")


class TemporalBranch(nn.Module):
    """Shared temporal component of CKNet (module docstring)."""

    def __init__(self, cfg: TemporalCfg, d_model: int = 192, seed: int = 0, ca_b: Optional[Tensor] = None):
        super().__init__()
        if not cfg.on:
            raise ValueError("TemporalBranch needs cfg.on (off = no branch at all)")
        self.cfg = cfg
        self.d_model = int(d_model)
        self.roi = cfg.roi
        self.check_inputs = True
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + TEMPORAL_SEED_OFFSET)
            c1, c2, c3 = ST_CH
            self.st_enc = nn.Sequential(
                nn.Conv2d(C_R, c1, 3, padding=1), nn.GroupNorm(4, c1), nn.GELU(),
                nn.Conv2d(c1, c2, 3, padding=1), nn.GroupNorm(8, c2), nn.GELU(),
                nn.Conv2d(c2, c3, 3, padding=1), nn.GroupNorm(8, c3), nn.GELU())
            self.st_proj = nn.Linear(c3 * N_RT_TOK, self.d_model)
            self.gap_proj = nn.Linear(N_GAP, self.d_model)
            for m in (self.st_proj, self.gap_proj):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        if cfg.forecaster == "learned":
            self.forecaster = LearnedForecaster(n_modes=cfg.n_modes, seed=int(seed))
        else:
            self.forecaster = None
            if ca_b is None:
                ca_b = load_ca_b(cfg.ca_scales, source=cfg.source)
            ca_b = torch.as_tensor(ca_b, dtype=torch.float32)
            if ca_b.shape != (2, FC_STEPS):
                raise ValueError(f"ca_b must be [2, {FC_STEPS}], got {tuple(ca_b.shape)}")
            self.register_buffer("ca_b", ca_b.clone(), persistent=True)

    # ------------------------------------------------------------------------------------------ bookkeeping
    def shared_parameters(self):
        for m in (self.st_enc, self.st_proj, self.gap_proj):
            yield from m.parameters()

    def param_counts(self) -> Dict[str, int]:
        return {"temporal_shared": sum(p.numel() for p in self.shared_parameters()),
                "forecaster": 0 if self.forecaster is None else sum(p.numel() for p in self.forecaster.parameters())}

    # ------------------------------------------------------------------------------------------ scene
    def scene(self, objs: Dict[str, Tensor]) -> TemporalCtx:
        missing = [k for k in OBJ_INPUT_KEYS if k not in objs]
        if missing:
            raise ValueError(f"temporal objs missing {missing}")
        bad = sorted(k for k in objs if k in TARGET_KEYS)
        if bad:
            raise ValueError(f"forecast targets {bad} must never reach forward()")
        oracle = self.cfg.forecaster == "oracle_gt"
        priv = [k for k in PRIV_INPUT_KEYS if k in objs]
        if priv and not oracle:
            raise ValueError(f"privileged keys {priv} given to a non-oracle temporal branch")
        if oracle and len(priv) != len(PRIV_INPUT_KEYS):
            raise ValueError(f"oracle forecaster needs {PRIV_INPUT_KEYS}")
        unknown = sorted(set(objs) - set(OBJ_INPUT_KEYS) - set(PRIV_INPUT_KEYS))
        if unknown:
            raise ValueError(f"unexpected temporal objs keys {unknown}")
        obj_in = objs["obj_in"]
        dev = obj_in.device
        with torch.autocast(device_type=dev.type, enabled=False):
            obj_in = obj_in.float()
            obj_mask = objs["obj_mask"].bool()
            frame_valid = objs["frame_valid"].bool()
            if self.check_inputs:
                check_causal(obj_in, obj_mask, frame_valid)
            obj_in, obj_mask, frame_valid = trim_objects(obj_in, obj_mask, frame_valid)
            fc_raw = None
            if self.cfg.forecaster == "learned":
                if self.cfg.fc_frozen:          # pre-trained + frozen (review fix D3): a fixed forecaster, as CA
                    with torch.no_grad():
                        fc_raw = self.forecaster(obj_in, obj_mask, frame_valid)
                    fc = fc_raw
                else:
                    fc_raw = self.forecaster(obj_in, obj_mask, frame_valid)
                    fc = fc_raw if self.cfg.fc_backprop else fc_raw.detach()
            elif self.cfg.forecaster == "ca":
                fc = ca_forecast(obj_in, obj_mask, frame_valid, self.ca_b)
            else:
                fc = oracle_forecast(obj_in, obj_mask, frame_valid, objs["priv_fc_xyh"].to(dev).float(),
                                     objs["priv_fc_state"].to(dev), self.ca_b)
            fc = trim_forecast(fc)
        return TemporalCtx(fc=fc, fc_raw=fc_raw, n_obj=int(fc.mu.shape[1]), n_valid=int(fc.valid.sum()))

    # ------------------------------------------------------------------------------------------ per candidate set
    def raster_and_gap(self, ctx: TemporalCtx, geom, flat_tau: Tensor, T: int, Kc: int) -> Tuple[Tensor, Tensor]:
        """R [B, 18, 48, 21], G [B, 20] (float32, autocast off; no_grad unless cfg.fc_backprop)."""
        grad = bool(self.cfg.fc_backprop and self.forecaster is not None and torch.is_grad_enabled())
        with torch.set_grad_enabled(grad), torch.autocast(device_type=flat_tau.device.type, enabled=False):
            R = rasterize(ctx.fc, geom, T, Kc, self.roi)
            G = gap_scalars(ctx.fc, flat_tau.float(), T, Kc)
        return R, G

    def features(self, ctx: TemporalCtx, geom, flat_tau: Tensor, T: int, Kc: int) -> Tuple[Tensor, Tensor]:
        """-> st_add [B, 48, d], ud_add [B, d] (st_enc / projections under the caller's autocast)."""
        R, G = self.raster_and_gap(ctx, geom, flat_tau, T, Kc)
        B = R.shape[0]
        H = self.st_enc(R)                                                            # [B, 48, 48, 21]
        st = H.permute(0, 2, 1, 3).reshape(B, N_ST_TOK, -1)                           # [B, 48 stations, 1008]
        return self.st_proj(st), self.gap_proj(G)

    def hook(self, ctx: TemporalCtx, T: int, Kc: int) -> Callable:
        """tfeat(geom, flat) bound to ctx and this candidate set's Kc (CKTrunk.candidates integration)."""
        def tfeat(geom, flat):
            if flat.shape[0] != T * Kc:
                raise ValueError(f"tfeat bound to T={T}, Kc={Kc} got {flat.shape[0]} candidates")
            return self.features(ctx, geom, flat, T, Kc)
        return tfeat

    def extra_repr(self) -> str:
        c = self.param_counts()
        return (f"forecaster={self.cfg.forecaster}, n_modes={self.cfg.n_modes}, fc_backprop={self.cfg.fc_backprop}, "
                f"temporal_shared={c['temporal_shared']}, forecaster_params={c['forecaster']}")
