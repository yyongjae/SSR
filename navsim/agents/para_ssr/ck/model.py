"""CK (Corridor candidate head) network: v1 RefinerNet trunk reused + per-candidate score head + v1 correction branch.

Inputs (T tokens x K candidates; candidate b = t * K + k)
  bev    : [T, 256, 50, 100] S grid (row = x forward 0.64 m, col 0 = 32 m LEFT), f16 or f32.
           arm 'S': v2 bev_embed reshaped (bev_embed.T.reshape(256, 50, 100)); 'T': BEVFusion t0 cache
           (load_bev(s_grid=True)); 'M': ReSMap cache (load_bev(s_grid=True)).
  cand   : [T, K, 8, 3] f32 executed candidates in the N frame (x fwd, y left, heading; t = 0.5 .. 4.0 s). Input only.
  status : [T, 8] f32 = cmd one-hot (left, straight, right, unknown) + vx, vy, ax, ay -> refiner.e2e.ego_inputs.
  extra  : optional [T, K2, 8, 3] further candidates scored with the SAME scene features (no second adapter pass).
Outputs (CKNet.forward dict)
  score_logit [T, K, 5] f32 (CK_KEYS = nc, dac, ep, ttc, comfort), z_lon / w_lat [T, K, 6] f32 (raw decoder
  controls, zero-initialised heads -> identity), corr (correct.correct dict, if decode), extra_score_logit [T, K2, 5],
  corr_score_logit [T, K, 5] (rescore_corr: CK scores of its own detached corrected candidates), lead_logit [T]
  (lead_aux), gate_logit [T, K] (v1 gate head on the detached draft token; unused), ego (v0, a0, eds, cmd).

Structure
  CKTrunk(RefinerNet): RefinerNet.forward split into scene() (adapter + global memory, once per token) and
  candidates() (corridor + draft token + 4 layers + heads, per candidate set).  Same parameters / state-dict keys as
  RefinerNet, so v1 stage-T checkpoints load directly; forward() is bit-identical to RefinerNet.forward.
  score_head: Linear(384, 256)-GELU-Linear(256, 5) on h = [x_draft || mean stations] (NOT detached: score labels train
  the trunk).  Built under torch.manual_seed(seed + SCORE_SEED_OFFSET) so every arm starts from the same head.
  arm 'S' adapter: AdapterSGrid = refiner.e2e.AdapterS (per-cell LayerNorm, 1x1 256-128-64) on the S-grid layout
  (same keys / init as e2e.build_student).
  Geometry / draft features / decoder in float32 outside autocast (as RefinerNet); the trunk under the caller's autocast.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ..refiner.adapters import BEV_H, BEV_W, load_norm
from ..refiner.corridor import CORRIDOR_VERSIONS, N_ST, corridor_geometry, corridor_tensor, global_tokens
from ..refiner.decoder import N_FREE
from ..refiner.e2e import AdapterS, ego_inputs
from ..refiner.geometry import N_POSE
from ..refiner.refiner_net import ADAPTER_SEED_OFFSET, RefinerNet, draft_features
from .constants import CK_KEYS, NORM_FILE, SCORE_SEED_OFFSET
from .correct import correct
from .temporal.config import TemporalCfg
from .v4.config import V4Cfg
from .v5.config import V5Cfg

ARMS = ("T", "M", "S", "none")
N_KEYS = len(CK_KEYS)
V4_SEED_OFFSET = 40414      # refiner v4 heads: own seeded RNG, built after every CK2 / v3 module
V5_SEED_OFFSET = 50515      # refiner v5 modules (ck/v5): own seeded RNG, built after every v4 module
PROBE_SPEEDS, PROBE_LATS = (-0.5,), (-0.5, 0.5)   # refiner v5 top_feats 'v2' Delta r probes (design §3.7)
GEO_T_ARR, GEO_V_ARR = 64 + 4, 64 + 5    # corridor_tensor channels of t_arr / v_arr (64 features + GEO_CHANNELS)


# ----------------------------------------------------------------------------------------------- adapter (student)
class AdapterSGrid(AdapterS):
    """AdapterS on the S-grid layout: bev [T, 256, 50, 100] -> AdapterS.forward(bev.flatten(2).transpose(1, 2))
    (index row * 100 + col = v2 bev_embed index) -> [T, 64, 50, 100].  Stateless per-cell LayerNorm."""

    def forward(self, bev: torch.Tensor, n_tokens: Optional[int] = None) -> torch.Tensor:
        if bev is None or bev.dim() != 4 or tuple(bev.shape[1:]) != (self.in_ch, BEV_H, BEV_W):
            raise ValueError(f"AdapterSGrid needs bev [T, {self.in_ch}, {BEV_H}, {BEV_W}], got "
                             f"{None if bev is None else tuple(bev.shape)}")
        return super().forward(bev.flatten(2).transpose(1, 2), n_tokens)


# ----------------------------------------------------------------------------------------------- trunk
class CKTrunk(RefinerNet):
    """RefinerNet with forward split into scene() / candidates() (same parameters and keys)."""

    def scene(self, bev: Optional[torch.Tensor], n_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """-> feat [T, 64, 50, 100], mem [T, 200, d] (global tokens + learned positions)."""
        feat = self.adapter(bev, n_tokens=n_tokens)
        mem = self.global_proj(global_tokens(feat)) + self.global_pos
        return feat, mem

    def candidates(self, feat: torch.Tensor, mem: torch.Tensor, tau: torch.Tensor, v0: torch.Tensor,
                   a0: torch.Tensor, eds: torch.Tensor, cmd: torch.Tensor,
                   tfeat: Optional[Callable] = None) -> Dict[str, torch.Tensor]:
        """tau [T, K, 8, 3] -> h [T*K, 2d] (not detached), z_lon / w_lat [T, K, 6], gate_logit [T, K], ud, geom.
        tfeat (refiner-v3 temporal branch, TemporalBranch.hook): (geom, flat) -> (st_add [B, 48, d], ud_add [B, d]),
        added to the 48 station tokens and the draft token; None = the CK2 path (required None when off)."""
        T, K = tau.shape[:2]
        B = T * K
        with torch.autocast(device_type=tau.device.type, enabled=False):
            flat = tau.reshape(B, N_POSE, 3).float()
            geom = corridor_geometry(flat, version=self.corridor_version)
            rep = lambda x: x.reshape(T, *x.shape[1:]).repeat_interleave(K, 0)
            ud = draft_features(flat, rep(v0.float()), rep(a0.float()), rep(eds.float()), rep(cmd), geom)
        X = corridor_tensor(feat, geom, T, K)                                         # [B, 70, 48, 17]
        Hc = self.enc(X)                                                              # [B, 128, 48, 17]
        st = self.station(Hc.permute(0, 2, 1, 3).reshape(B, N_ST, -1))               # [B, 48, d]
        if tfeat is None:
            if getattr(self, "_temporal_required", False):
                raise RuntimeError("CKTrunk.candidates: this CK net has a temporal branch; call it through "
                                   "CKNet.forward(..., objs=...) (direct trunk use would silently drop it)")
            x = torch.cat([self.draft_mlp(ud.to(st.dtype))[:, None], st], 1)          # [B, 49, d]
        else:
            st_add, ud_add = tfeat(geom, flat)
            x = torch.cat([(self.draft_mlp(ud.to(st.dtype)) + ud_add.to(st.dtype))[:, None],
                           st + st_add.to(st.dtype)], 1)                              # [B, 49, d]
        for layer in self.layers:
            x = layer(x, mem, T, K)
        x = self.norm_out(x)
        xd = x[:, 0]
        gate = self.gate_head(xd.detach())[:, 0]
        h = torch.cat([xd, x[:, 1:].mean(1)], -1)
        return {"h": h, "gate_logit": gate.reshape(T, K), "z_lon": self.lon_head(h).reshape(T, K, N_FREE),
                "w_lat": self.lat_head(h).reshape(T, K, N_FREE), "ud": ud, "geom": geom}

    def stations(self, feat: torch.Tensor, mem: torch.Tensor, tau: torch.Tensor, v0: torch.Tensor,
                 a0: torch.Tensor, eds: torch.Tensor, cmd: torch.Tensor, time_free: bool = True
                 ) -> Dict[str, torch.Tensor]:
        """refiner v4 parent pass: tau [T, K, 8, 3] (parent drafts) -> x [T*K, 49, d] (draft token + 48 station tokens
        after the 4 layers and norm_out), Hc [T*K, 128, 48, 17] (corridor conv features), ud, geom.  time_free zeroes
        the t_arr / v_arr geo channels (the station features then carry no schedule; candidates add their own).
        Same modules as candidates() (no heads).
        Row independence (CKNet.score_set's parent dedupe relies on it): every op is per draft row -- geometry, the
        per-token grid_sample, Conv / GroupNorm (per sample) / station Linear, self-attention within the row's 49
        tokens, cross-attention per query to its TOKEN's 200 global tokens (the K rows' queries of a token are only
        concatenated, softmax is over the keys), FFN / LayerNorm, dropout 0 -- so a row's output does not depend on
        which other rows (or how many, K) are in the set.  Measured bit for bit on CPU fp32 (tests/test_v4_dedupe.py);
        a GPU kernel choice may still depend on the batch size (rounding only)."""
        T, K = tau.shape[:2]
        B = T * K
        with torch.autocast(device_type=tau.device.type, enabled=False):
            flat = tau.reshape(B, N_POSE, 3).float()
            geom = corridor_geometry(flat, version=self.corridor_version)
            rep = lambda x: x.reshape(T, *x.shape[1:]).repeat_interleave(K, 0)
            ud = draft_features(flat, rep(v0.float()), rep(a0.float()), rep(eds.float()), rep(cmd), geom)
        X = corridor_tensor(feat, geom, T, K)                                         # [B, 70, 48, 17]
        if time_free:
            X = X.clone()
            X[:, GEO_T_ARR] = 0
            X[:, GEO_V_ARR] = 0
        Hc = self.enc(X)                                                              # [B, 128, 48, 17]
        st = self.station(Hc.permute(0, 2, 1, 3).reshape(B, N_ST, -1))               # [B, 48, d]
        x = torch.cat([self.draft_mlp(ud.to(st.dtype))[:, None], st], 1)              # [B, 49, d]
        for layer in self.layers:
            x = layer(x, mem, T, K)
        return {"x": self.norm_out(x), "Hc": Hc, "ud": ud, "geom": geom}

    def forward(self, bev, tau0, v0, a0, eds, cmd) -> Dict[str, torch.Tensor]:
        feat, mem = self.scene(bev, tau0.shape[0])
        return self.candidates(feat, mem, tau0, v0, a0, eds, cmd)


def build_trunk(arm: str, seed: int = 0, norm: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> CKTrunk:
    """arm 'T' (BEVFusion z-score norm), 'M' (ReSMap norm), 'S' (AdapterSGrid, build_student pattern), 'none'.
    No global RNG is consumed."""
    if arm not in ARMS:
        raise ValueError(f"arm {arm!r} not in {ARMS}")
    mean, std = norm if norm is not None else (None, None)
    with torch.random.fork_rng(devices=[]):
        if arm == "T":
            tr = CKTrunk("T", mean, std, seed=seed)
        elif arm == "M":
            tr = CKTrunk("M", seed=seed, map_norm_mean=mean, map_norm_std=std)
        else:
            tr = CKTrunk("none", seed=seed)
            if arm == "S":
                torch.manual_seed(seed + ADAPTER_SEED_OFFSET)
                tr.adapter = AdapterSGrid()
                tr.arm = "S"
    return tr


# ----------------------------------------------------------------------------------------------- CK net
class CKNet(nn.Module):
    """CK head (module docstring).  CKNet(arm in {'T', 'M', 'S'}, seed, norm=(mean, std) for T / M)."""

    def __init__(self, arm: str, seed: int = 0, norm: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 score_hidden: int = 256, lead_aux: bool = False, temporal: Optional[TemporalCfg] = None,
                 ca_b: Optional[torch.Tensor] = None, corridor_version: int = 1, v4: Optional[V4Cfg] = None,
                 v5: Optional[V5Cfg] = None):
        super().__init__()
        self.arm = arm
        self.seed = int(seed)
        self.lead_aux = bool(lead_aux)
        self.trunk = build_trunk(arm, seed, norm)
        # refiner-fix B1: corridor geometry version (plain attribute, not state; config.json 'corridor_version')
        if int(corridor_version) not in CORRIDOR_VERSIONS:
            raise ValueError(f"corridor_version {corridor_version!r} not in {CORRIDOR_VERSIONS}")
        self.corridor_version = int(corridor_version)
        self.trunk.corridor_version = self.corridor_version
        d2 = 2 * self.trunk.d_model
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + SCORE_SEED_OFFSET)
            self.score_head = nn.Sequential(nn.Linear(d2, score_hidden), nn.GELU(), nn.Linear(score_hidden, N_KEYS))
            self.lead_head = (nn.Sequential(nn.Linear(d2, 128), nn.GELU(), nn.Linear(128, 1)) if lead_aux else None)
        self.init_report: Dict = {}
        # refiner-v3 temporal branch (docs/refiner_v3/SPEC.md 4.5): built AFTER every CK2 module, own seeded RNG
        self.temporal_cfg = temporal if temporal is not None else TemporalCfg()
        if self.temporal_cfg.on:
            self.temporal_cfg.validate(arm)
            from .temporal.branch import TemporalBranch
            self.temporal = TemporalBranch(self.temporal_cfg, self.trunk.d_model, seed, ca_b=ca_b)
            self.trunk._temporal_required = True                       # plain attribute, not state
        else:
            self.temporal = None
        # refiner v4 (ck/v4): built AFTER every CK2 / v3 module under its own seed (V4Cfg() = off: nothing is built,
        # the state dict and every output are the CK2 / v3 ones)
        self.v4 = (v4 if v4 is not None else V4Cfg()).validate(self.temporal_cfg.on)
        self.v4_readout = None
        self.pref_head = None
        self.corrector = None
        if self.v4.on:
            from .v4.evidence import EvidenceReadout
            from .v4.readout import HazardReadout, PRefHead
            if self.v4.parent_pass and self.lead_aux:
                raise ValueError(f"v4 {self.v4.readout} readout: lead_aux is not supported")
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + V4_SEED_OFFSET)
                if self.v4.hazard:
                    self.v4_readout = HazardReadout(self.v4, self.trunk.d_model)
                elif self.v4.evidence:
                    self.v4_readout = EvidenceReadout(self.v4, self.trunk.d_model)
                elif self.v4.range:
                    from .v4.range_readout import RangeReadout
                    self.v4_readout = RangeReadout(self.v4, self.trunk.d_model)
                if self.v4.pref_head:
                    self.pref_head = PRefHead(self.trunk.d_model, kdim=self.v4.pref_kdim, hidden=self.v4.pref_hidden)
                if self.v4.corrector:
                    from .v4.corrector import Corrector
                    self.corrector = Corrector(self.trunk.d_model)
            if self.v4.parent_pass:
                # the CK2 heads the hazard readout replaces: kept in the state dict (loaders stay uniform) but frozen,
                # so DDP / optimisers never expect a gradient for them
                for mod in (self.score_head, self.trunk.lat_head, self.trunk.lon_head, self.trunk.gate_head):
                    mod.requires_grad_(False)
        # refiner v5 (ck/v5): built AFTER every v4 module under its own seed (V5Cfg() = off: nothing is built)
        self.v5 = (v5 if v5 is not None else V5Cfg()).validate(self.v4)
        self.relation = None
        self.corrector_v5 = None
        self.z_adapter = None
        if self.v5.on:
            from .v5.config import N_TOP
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + V5_SEED_OFFSET)
                self.v4_readout.enable_v5(self.v5.n_extra, self.v5.rel, self.v5.unknown_ev, self.v5.z_stride)
                if self.v5.rel:
                    from .v5.relation import RelationReader
                    self.relation = RelationReader(self.v5)
                if self.v5.corrector:
                    from .v5.corrector import CorrectorV5
                    # top-3 object features: N_TOP_FEAT ('v1', the teachers' A_T) or 13 ('v2': + station / side / key
                    # + Delta r of the probes, V5Cfg.top_feats)
                    self.corrector_v5 = CorrectorV5(self.trunk.d_model, self.v5.n_z,
                                                    N_TOP * self.v5.n_top_feat if self.v5.rel else 0,
                                                    hidden=self.v5.corr_hidden, z_hidden=self.v5.z_hidden)
                if self.v5.z_adapt:
                    # student Z adapters P_det / P_map (ck/v5/zadapt.py): built LAST in the v5 fork, so every other v5
                    # module's init is bit for bit the no-adapter net's; only the half z_channels keeps
                    from .v5.zadapt import ZAdapter
                    self.z_adapter = ZAdapter(self.v5.n_z, self.v5.z_adapt_hidden, self.v5.z_channels)

    # ------------------------------------------------------------------------------------------ helpers
    def param_counts(self) -> Dict[str, int]:
        c = self.trunk.param_counts()
        c["score_head"] = sum(p.numel() for p in self.score_head.parameters())
        c["lead_head"] = 0 if self.lead_head is None else sum(p.numel() for p in self.lead_head.parameters())
        c["total"] = c["total"] + c["score_head"] + c["lead_head"]
        if self.temporal is not None:                                   # keys only when on (OFF dict unchanged)
            tc = self.temporal.param_counts()
            c.update(tc)
            c["total"] = c["total"] + tc["temporal_shared"] + tc["forecaster"]
        for name, mod in (("v4_readout", self.v4_readout), ("pref_head", self.pref_head),
                          ("corrector", self.corrector), ("relation", getattr(self, "relation", None)),
                          ("corrector_v5", getattr(self, "corrector_v5", None)),
                          ("z_adapter", getattr(self, "z_adapter", None))):
            if mod is not None:                                         # keys only when on
                c[name] = sum(p.numel() for p in mod.parameters())
                c["total"] = c["total"] + c[name]
        return c

    @torch.no_grad()
    def set_score_prior(self, prior: Sequence[float]) -> None:
        """Set the score head's output bias to logit(prior) per key (CK_KEYS order), e.g. label means."""
        p = torch.as_tensor(np.asarray(prior, np.float64)).clamp(1e-4, 1 - 1e-4)
        self.score_head[-1].bias.copy_(torch.log(p / (1 - p)).to(self.score_head[-1].bias))
        if self.v4_readout is not None:
            self.v4_readout.set_prior(np.asarray(prior, np.float64))

    def score(self, cand_out: Dict[str, torch.Tensor], T: int, K: int) -> torch.Tensor:
        return self.score_head(cand_out["h"]).float().reshape(T, K, N_KEYS)

    def scene_pref(self, mem: torch.Tensor, ego) -> Optional[torch.Tensor]:
        """v4 pref_head: log1p(P_ref) [T] (None when the head is off)."""
        if self.pref_head is None:
            return None
        v0, a0, _, cmd = ego
        return self.pref_head(mem, v0, a0, cmd)

    # refiner v4 parent dedupe (score_set): run trunk.stations once per unique parent of a token (plain attribute,
    # not state; score_set(dedupe=...) overrides it per call, False = the repeated per-candidate parent pass)
    dedupe_parents = True

    @staticmethod
    def unique_parents(par: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-token unique parent drafts by BITWISE equality of the float32 values (equal rows are equal inputs:
        a NaN row matches its own bit pattern, -0.0 and +0.0 differ): par [T, K, 8, 3] ->
        par_u [T, U, 8, 3] f32 (first-occurrence order, U = max_t U_t; slots u >= U_t repeat the token's first parent:
        computed, never read), parent_index [T, K] int64 (par[t, k] == par_u[t, parent_index[t, k]] bit for bit),
        n_unique [T] int64.  O(K^2) compares per token (K <= a few hundred here)."""
        T, K = par.shape[:2]
        p = par.float().contiguous()
        key = p.view(torch.int32).reshape(T, K, -1)                                   # [T, K, 24] bit patterns
        eq = torch.ones(T, K, K, dtype=torch.bool, device=p.device)
        for c in range(key.shape[-1]):                                                # peak memory T K^2 (not x24)
            eq &= key[:, :, None, c] == key[:, None, :, c]
        ar = torch.arange(K, device=p.device)
        first = torch.where(eq, ar, torch.full_like(ar, K)).amin(-1)                  # [T, K] first equal row (<= k)
        is_u = first == ar
        rank = torch.cumsum(is_u.long(), 1) - 1                                       # slot of each first occurrence
        pidx = torch.gather(rank, 1, first)
        n_u = is_u.sum(1)
        U = int(n_u.max()) if T * K else 0
        slot = torch.where(is_u, rank, torch.full_like(rank, U))                      # repeats -> dump column U
        src = torch.zeros(T, U + 1, dtype=torch.long, device=p.device)
        src = src.scatter(1, slot, ar.expand(T, K).contiguous())[:, :U]               # pad slots -> row 0 (unique)
        par_u = torch.gather(p, 1, src[..., None, None].expand(T, U, *p.shape[2:]))
        return par_u, pidx, n_u

    def score_set(self, feat: torch.Tensor, mem: torch.Tensor, ego, cand: torch.Tensor,
                  parents: Optional[torch.Tensor] = None, tfeat=None, pref: Optional[torch.Tensor] = None,
                  return_ctx: bool = False, parent_index: Optional[torch.Tensor] = None, pctx: Optional[Dict] = None,
                  return_pctx: bool = False, dedupe: Optional[bool] = None,
                  priv: Optional[Dict[str, torch.Tensor]] = None,
                  rel_objs: Optional[Dict[str, torch.Tensor]] = None, zq_exec: Optional[str] = None) -> Dict:
        """Score one candidate set with the token's scene (feat, mem from trunk.scene).  zq_exec (refiner v5, range +
        v5 nets; a KD teacher of the no-LQR student): an exec model other than v4.exec_model adds haz['zq_kd'] = the
        per-state reads zq (+ the relation's H channels) of the same fields at THOSE states of the candidates (no
        scoring; every other output unchanged); None / the own exec model: no zq_kd.  priv: the batch's ref_obj_*
        dict (losses.EV_REF_KEYS) of a PRIVILEGED range teacher (V4Cfg.rg_priv != 'off'; required then, refused
        otherwise; range_targets.priv_channels on the parent rows, reused from a range pctx):
        cand [T, K, 8, 3] -> score_logit [T, K, 5] f32, w_lat / z_lon [T, K, 6] f32, gate_logit [T, K], and for the v4
        parent-pass readouts 'haz' = {t_arr (bin-centre first arrival), t_in / t_out (entry / min(leave, horizon) of
        the station bins), d_lat, haz_mask [T, K, 48], is_self [T, K]; hazard: a_nc / a_ttc / a_dac
        [T, K, 48]; evidence: occ_logit [T, K, 48, 17, M], margin [T, K, 48, 17], pts [T, K, 48, 17, 2] (parent grid
        points, N frame), lp_nc / lp_ttc / lp_dac / min_margin [T, K]; range (V4_3 M4, ck/v4/range_readout.py):
        occ_logit [T, K, 48, 17, M] (B of the candidate's parent), margin [T, K, 48, 17] (D), the same once per
        parent row occ_logit_u / ovl_logit_u (U) [T, U, 48, 17, M] (+ bp_logit_u (Bp, the plain not-behind B that
        TTC reads) with v4.rg_stopped_aware), margin_u [T, U, 48, 17] with parent_index
        [T, K] (row of each candidate; U = K and index k without dedupe), exec_states [T, K, Q, 6] (Q 41 'lqr' / 'ref',
        9 'plan'), lp_nc / lp_ttc / lp_dac, slack_nc / slack_ttc [s] / slack_dac (= min_margin) [m], exec_ratio,
        S_exec [m], overspeed [m/s] [T, K], crit_station / crit_lat (long) / crit_time [T, K, 3] (nc, ttc, dac),
        crit_chan [T, K], pts}.
        readout 'mean': exactly trunk.candidates + score (the CK2 / v3 ops; parents and every v4 argument ignored).
        readout 'hazard' / 'evidence' / 'range': parents [T, K, 8, 3] = each candidate's parent draft (None: every
        candidate is its own parent); the scene is read on the parent's corridor, the candidate reads it at its own
        schedule (range: at its executed states).
        Parent dedupe (spec V4_3 M3): trunk.stations (the parent pass: corridor conv + 4 layers, ~0.64 GFLOP per row;
        99 % of the repeated hazard score_set FLOPs, 91 % of evidence, whose O / D scene head still runs per candidate)
        runs once per UNIQUE parent of each token and its x / Hc rows are gathered back to the candidates; the
        candidates' own quantities (on_parent, corridor geometry of the parent rows, draft features, readout) stay per
        candidate (96 pool, 16 parents: 5.7x fewer FLOPs).  Bit for bit the repeated pass (row independence, see
        trunk.stations; max diff 0 on CPU fp32, tests/test_v4_dedupe.py); gradients equal up to summation order (the
        K rows' gradients are summed before the parent pass instead of inside its weight GEMMs).
          * parents [T, K, 8, 3]: unique rows found by bitwise equality (unique_parents); dedupe (None =
            self.dedupe_parents, default True) False, or some token without a repeated parent (U = K, no saving) ->
            the repeated per-candidate pass.
          * parent_index [T, K] int64 + parents [T, U, 8, 3] (the unique parents): candidate k's parent is row
            parent_index[t, k] (no search; the training sets know their parents' columns).
          * pctx (+ parent_index; parents None or pctx['parents']): the per-parent context of an earlier
            score_set(..., return_pctx=True) with the SAME feat / mem / ego -> no parent pass at all (the corrector's
            edits re-scored on their sources' parents, correct_topk / corrector_logp).  Gradients flow into that
            earlier graph (none if it ran under no_grad).
        return_pctx -> + 'pctx' = {parents [T, U, 8, 3] f32, x [T*U, 49, d], Hc [T*U, 128, 48, 17], ud [T*U, 69],
        geom (corridor_geometry of the T*U rows); range: + rg_fields (RangeReadout.scene of the rows), rg_frame
        (range_targets.parent_frame of the rows)} and 'parent_index' [T, K] (U = K, index = k without dedupe)."""
        T, K = cand.shape[:2]
        v0, a0, eds, cmd = ego
        if self.v4_readout is None:
            o = self.trunk.candidates(feat, mem, cand, v0, a0, eds, cmd, tfeat=tfeat)
            return {"score_logit": self.score(o, T, K), "w_lat": o["w_lat"].float(), "z_lon": o["z_lon"].float(),
                    "gate_logit": o["gate_logit"].float(), "h": o["h"]}
        if tfeat is not None:
            raise ValueError("v4 hazard readout has no temporal hook")
        from .v4.readout import feasibility
        from .v4.sibling import on_parent
        B = T * K
        cand = cand.float()
        dev = cand.device
        own = parents is None and parent_index is None                  # every candidate is its own parent
        if pctx is not None and parent_index is None:
            raise ValueError("score_set: pctx needs parent_index (each candidate's row of pctx['parents'])")
        par_u = pidx = P = None
        if parent_index is not None:                                    # explicit unique parents (+ reused context)
            if pctx is not None:
                par_u = pctx["parents"]
                if parents is not None and not (tuple(parents.shape) == tuple(par_u.shape) and torch.equal(
                        parents.float().contiguous().view(torch.int32), par_u.contiguous().view(torch.int32))):
                    raise ValueError("score_set: parents differ from pctx['parents']")
                P = pctx
            elif parents is None:
                raise ValueError("score_set: parent_index needs parents [T, U, 8, 3] (the unique parents) or pctx")
            else:
                par_u = parents.float()
            pidx = parent_index.to(device=dev, dtype=torch.long)
            U = par_u.shape[1]
            if (par_u.dim() != 4 or par_u.shape[0] != T or tuple(pidx.shape) != (T, K)
                    or bool(((pidx < 0) | (pidx >= U)).any())):
                raise ValueError(f"score_set: parent_index {tuple(pidx.shape)} (range [0, {U})) / parents "
                                 f"{tuple(par_u.shape)} vs cand {tuple(cand.shape)}")
            par = torch.gather(par_u, 1, pidx[..., None, None].expand(T, K, *par_u.shape[2:]))
        else:
            par = cand if parents is None else parents.float()
            if tuple(par.shape) != tuple(cand.shape):
                raise ValueError(f"score_set: parents {tuple(par.shape)} vs cand {tuple(cand.shape)}")
            if not own and (self.dedupe_parents if dedupe is None else dedupe) and K > 1 and T > 0:
                par_u, pidx, _ = self.unique_parents(par)
                if par_u.shape[1] == K:                                 # a token without repeats: no saving
                    par_u = pidx = None
        if pidx is None:                                                # the repeated per-candidate parent pass
            P = self.trunk.stations(feat, mem, par, v0, a0, eds, cmd, time_free=self.v4.time_free)
            x, Hc, pgeom = P["x"], P["Hc"], P["geom"]
            rows = None                                                 # P row b = candidate b
        else:                                                           # once per unique parent, gathered back
            if P is None:
                P = self.trunk.stations(feat, mem, par_u, v0, a0, eds, cmd, time_free=self.v4.time_free)
            rows = (torch.arange(T, device=dev)[:, None] * par_u.shape[1] + pidx).reshape(B)
            x = P["x"].index_select(0, rows)
            # range: Hc is read on the parent rows (range_readout.lateral_read_rows), no per-candidate copy
            Hc = None if self.v4.range else P["Hc"].index_select(0, rows)
            with torch.autocast(device_type=dev.type, enabled=False):  # = the repeated pass's parent geometry
                pgeom = corridor_geometry(par.reshape(B, N_POSE, 3), version=self.corridor_version)
        flat = cand.reshape(B, N_POSE, 3)
        rep = lambda x: x.reshape(T, *x.shape[1:]).repeat_interleave(K, 0)
        with torch.autocast(device_type=cand.device.type, enabled=False):
            is_self = (flat == par.reshape(B, N_POSE, 3)).flatten(1).all(1)
            sib = on_parent(pgeom, flat, is_self)
            cgeom = pgeom if own else corridor_geometry(flat, version=self.corridor_version)
            ud = P["ud"] if own else draft_features(flat, rep(v0.float()), rep(a0.float()),
                                                    rep(eds.float()), rep(cmd), cgeom)
        g = self.trunk.draft_mlp(ud.to(x.dtype))
        pr = None if pref is None else rep(pref.reshape(T, 1).float())[:, 0]
        sh = lambda a: a.reshape(T, K, *a.shape[1:])
        rg = None
        if self.v4.hazard:
            e = feasibility(cgeom.S, rep(v0.float().reshape(T, 1))[:, 0])
            r = self.v4_readout(x[:, 1:], x[:, 0], Hc, g, sib["t_in"], sib["t_leave"], sib["v"], sib["d"],
                                cgeom.S[:, -1], e, pr)
            haz = {k: sh(r[k]) for k in ("a_nc", "a_ttc", "a_dac")}
        elif self.v4.evidence:
            r = self.v4_readout(x[:, 1:], x[:, 0], Hc, g, pgeom.s, sib["s_knots"], sib["d_knots"], sib["d"],
                                sib["t_in"], sib["t_leave"], cgeom.S[:, -1], pr)
            haz = {k: sh(r[k]) for k in ("occ_logit", "margin", "lp_nc", "lp_ttc", "lp_dac", "min_margin")}
            haz["pts"] = sh(pgeom.points.detach().float())
        else:                                                           # range (V4_3 M4): fields once per parent row
            r, haz, rg = self._range_read(P, rows, pidx, T, K, flat, ego, x, g, cgeom, sib, pr, priv, rel_objs,
                                          zq_exec=zq_exec)
            haz["pts"] = sh(pgeom.points.detach().float())
        haz.update(haz_mask=sh(r["mask"]), t_arr=sh(sib["t"]), t_in=sh(sib["t_in"]), t_out=sh(r["t_out"]),
                   d_lat=sh(sib["d"]), is_self=sh(is_self))
        out = {"score_logit": sh(r["score_logit"]), "w_lat": sh(r["w_lat"]),
               "z_lon": torch.zeros(T, K, N_FREE, dtype=torch.float32, device=cand.device),
               "gate_logit": torch.zeros(T, K, dtype=torch.float32, device=cand.device), "haz": haz}
        if return_ctx:          # corrector context (detached: the corrector never moves the critic's features)
            mf = r["mask"].to(x.dtype)[..., None]
            xs_mean = (x[:, 1:] * mf).sum(1) / mf.sum(1).clamp(min=1.0)
            out["ctx"] = sh(torch.cat([g, x[:, 0], xs_mean], -1).detach().float())
        if return_pctx:         # per-parent context for re-use (pctx=): the parent pass's own rows
            if pidx is None:
                pidx = torch.arange(K, device=dev)[None].expand(T, K)
                par_u = par
            out["pctx"] = P if pctx is not None else {"parents": par_u, "x": P["x"], "Hc": P["Hc"], "ud": P["ud"],
                                                      "geom": P["geom"]}
            if rg is not None and "rg_fields" not in out["pctx"]:       # range: the scene fields of the same rows
                out["pctx"] = dict(out["pctx"], rg_fields=rg[0], rg_frame=rg[1])
            out["parent_index"] = pidx
        return out

    def _range_read(self, P: Dict, rows: Optional[torch.Tensor], pidx: Optional[torch.Tensor], T: int, K: int,
                    flat: torch.Tensor, ego, x: torch.Tensor, g: torch.Tensor, cgeom, sib: Dict, pr,
                    priv: Optional[Dict[str, torch.Tensor]] = None,
                    rel_objs: Optional[Dict[str, torch.Tensor]] = None, zq_exec: Optional[str] = None):
        """score_set's readout 'range' (spec V4_3 M4; ck/v4/range_readout.py): scene fields on the R rows of the parent
        pass P (R = T * U unique parents, or T * K without dedupe; reused from P['rg_fields'] / P['rg_frame'] when P is
        a range pctx), the candidates' states (exec_model; ego (vx, ax) = eds[:, 0], eds[:, 2], i.e. status[..., 4] /
        status[..., 6], lqr docstring) read on their parent's row.  -> (readout dict, haz dict [T, K, ...],
        (fields, frame) of the R rows)."""
        from .v4 import range_targets as RT
        from .v4.range_readout import candidate_states
        B = T * K
        dev = flat.device
        prow = torch.arange(B, device=dev) if rows is None else rows
        if "rg_fields" in P:
            Fs, fr_u = P["rg_fields"], P["rg_frame"]
        else:
            fr_u = RT.parent_frame(P["geom"])
            pv = None
            if self.v4.rg_priv != "off":                     # PRIVILEGED input (V4Cfg.rg_priv, config docstring)
                if priv is None:
                    raise ValueError(f"CKNet.score_set: this range teacher is PRIVILEGED (v4.rg_priv "
                                     f"{self.v4.rg_priv!r}): priv= (the batch's ref_obj_* dict, losses.EV_REF_KEYS) "
                                     f"is required")
                R = int(P["Hc"].shape[0])
                tok_rows = torch.arange(R, device=dev) // max(R // T, 1)      # row r -> token (R = T * U rows)
                pv = RT.priv_channels(priv, tok_rows, fr_u, self.v4_readout.M, self.v4_readout.tbin,
                                      self.v4.rg_priv, device=dev)
            elif priv is not None:
                raise ValueError("CKNet.score_set: priv given to a range teacher without v4.rg_priv")
            Fs = (self.v4_readout.scene(P["x"][:, 1:], P["Hc"]) if pv is None       # off: the unchanged call
                  else self.v4_readout.scene(P["x"][:, 1:], P["Hc"], priv=pv))
        eds = ego[2].float()
        rep = lambda a: a.reshape(T, 1).repeat_interleave(K, 0)[:, 0]
        with torch.autocast(device_type=dev.type, enabled=False):
            states, tq = candidate_states(flat, rep(eds[:, 0]), rep(eds[:, 2]), self.v4.exec_model)
        v5in, rel = None, None
        if self.v5.on:                                   # refiner v5: relation channel on the strided executed states
            v5in = {}
            if self.relation is not None:
                if rel_objs is None:
                    raise ValueError("CKNet.score_set: this net has the v5 relation channel: rel_objs= (obj_in, "
                                     "obj_mask, frame_valid of the batch) is required")
                zs = int(self.v5.z_stride)
                tok = torch.arange(B, device=dev) // K
                rel = self.relation(rel_objs, states[:, ::zs], tq[::zs], tok)
                v5in = {"ev_extra": rel["ev"], "rel_lp": rel["lp"]}
        elif rel_objs is not None:
            raise ValueError("CKNet.score_set: rel_objs given to a net without the v5 relation channel")
        r = self.v4_readout(Fs, prow, fr_u, states, tq, x[:, 1:], x[:, 0], P["Hc"], g, cgeom.S[:, -1], sib["d"],
                            sib["t_in"], sib["t_leave"], pr, **({"v5": v5in} if v5in is not None else {}))
        M = self.v4_readout.M
        sh = lambda a: a.reshape(T, K, *a.shape[1:])
        R = Fs["blk"].shape[0]
        uq = lambda a: a.reshape(T, R // T, *a.shape[1:])
        per = lambda a: a if rows is None else a.index_select(0, rows)
        haz = {"occ_logit": sh(per(Fs["blk"][..., :M])), "margin": sh(per(Fs["margin"])),
               "occ_logit_u": uq(Fs["blk"][..., :M]), "ovl_logit_u": uq(Fs["blk"][..., M:2 * M]),
               "margin_u": uq(Fs["margin"]),
               "parent_index": (torch.arange(K, device=dev)[None].expand(T, K) if pidx is None else pidx),
               "exec_states": sh(states)}
        if Fs["blk"].shape[-1] == 3 * M:            # rg_stopped_aware: the plain not-behind B (TTC's B read)
            haz["bp_logit_u"] = uq(Fs["blk"][..., 2 * M:])
        haz.update({k: sh(r[k]) for k in ("lp_nc", "lp_ttc", "lp_dac", "slack_nc", "slack_ttc", "slack_dac",
                                           "min_margin", "crit_station", "crit_lat", "crit_time", "crit_chan",
                                           "exec_ratio", "S_exec", "overspeed")})
        if self.v5.on:                                   # refiner v5 outputs (per candidate; rel_a / rel_valid per token)
            zq = r["zq"] if rel is None else torch.cat([r["zq"], rel["H"].float()], -1)
            haz.update(zq=sh(zq), unknown_frac=sh(r["unknown_frac"]), corner_D=sh(r["corner_D"]),
                       corner_ok=sh(r["corner_ok"]))
            if rel is not None:
                haz.update(rel_lp=sh(rel["lp"]), rel_top=sh(rel["top"]), rel_r_nc=sh(rel["r_nc"]),
                           rel_r_ttc=sh(rel["r_ttc"]), rel_a=rel["a_tok"], rel_valid=rel["valid_tok"],
                           rel_objs_tok=rel["objs_tok"])
                if "top_idx" in rel:                     # V5Cfg.top_feats 'v2': object slot of each top entry (-1 none)
                    haz["rel_top_idx"] = sh(rel["top_idx"])
                if "lr_nc" in rel:                       # relation 'student': log r for the relation BCE (report 102 N1)
                    haz.update(rel_lr_nc=sh(rel["lr_nc"]), rel_lr_ttc=sh(rel["lr_ttc"]))
            if zq_exec is not None and zq_exec != self.v4.exec_model:   # Z-KD targets at the student's states
                with torch.autocast(device_type=dev.type, enabled=False):
                    st2, tq2 = candidate_states(flat, rep(eds[:, 0]), rep(eds[:, 2]), zq_exec)
                z2 = self.v4_readout.zq_at(Fs, prow, fr_u, st2, tq2)
                if self.relation is not None:
                    zs = int(self.v5.z_stride)
                    rel2 = self.relation(rel_objs, st2[:, ::zs], tq2[::zs], torch.arange(B, device=dev) // K)
                    z2 = torch.cat([z2, rel2["H"].float()], -1)
                haz["zq_kd"] = sh(z2)
        elif zq_exec is not None and zq_exec != self.v4.exec_model:
            raise ValueError("CKNet.score_set: zq_exec needs a v5 net (per-state reads)")
        return r, haz, (Fs, fr_u)

    def correct_topk(self, feat: torch.Tensor, mem: torch.Tensor, ego, pool: torch.Tensor, parents: torch.Tensor,
                     pool_out: Dict, pref: Optional[torch.Tensor] = None, sample: bool = False,
                     k: Optional[int] = None, valid: Optional[torch.Tensor] = None,
                     generator: Optional[torch.Generator] = None) -> Dict:
        """refiner v4 corrector A: shortlist the top-k of pool [T, N, 8, 3] by the critic selection score
        (corrector.critic_select_score of pool_out['score_logit'], invalid columns excluded), edit them (mu, or a
        sample when training), decode (mode B) and re-score the edits on their source's parent corridor.  pool_out
        = score_set(..., return_ctx=True) of the pool; with return_pctx=True as well, the edits re-use the pool's
        parent pass (pool_out 'pctx' / 'parent_index', no second trunk.stations; bit for bit the re-run, see
        score_set).  -> src_idx [T, k], u / mu / log_sig [T, k, 12], logp [T, k], edit_traj [T, k, 8, 3],
        edit_score_logit [T, k, 5], edit_w_lat [T, k, 6], edit_haz."""
        from .v4.corrector import critic_select_score, decode_edits
        if self.corrector is None:
            raise ValueError("correct_topk: this CK net has no corrector (V4Cfg.corrector)")
        T, N = pool.shape[:2]
        k = int(k or self.v4.corr_k)
        sc = critic_select_score(pool_out["score_logit"].detach())
        if valid is not None:
            sc = torch.where(valid.bool(), sc, torch.full_like(sc, -1e9))
        idx = torch.topk(sc, min(k, N), dim=1).indices                                 # [T, k]
        g4 = lambda a: torch.gather(a, 1, idx.reshape(T, -1, *([1] * (a.dim() - 2))).expand(T, idx.shape[1],
                                                                                                *a.shape[2:]))
        src, src_par = g4(pool.float()), g4(parents.float())
        ctx = torch.cat([g4(pool_out["ctx"]), g4(pool_out["score_logit"].detach().float()),
                         (ego[0].float() / 15.0).reshape(T, 1, 1).expand(T, idx.shape[1], 1)], -1)
        a = self.corrector.act(ctx, sample=sample, generator=generator)
        edits = decode_edits(src, a["u"], ego[0])
        if pool_out.get("pctx") is not None:            # the pool's parent pass (src_par = its rows, bit for bit)
            oe = self.score_set(feat, mem, ego, edits, pref=pref, parent_index=g4(pool_out["parent_index"]),
                                pctx=pool_out["pctx"])
        else:
            oe = self.score_set(feat, mem, ego, edits, src_par, pref=pref)
        return {"src_idx": idx, "u": a["u"], "mu": a["mu"], "log_sig": a["log_sig"], "logp": a["logp"],
                "edit_traj": edits, "edit_score_logit": oe["score_logit"], "edit_w_lat": oe["w_lat"],
                "edit_haz": oe.get("haz"), "ctx": ctx}

    def computed_edits(self, feat: torch.Tensor, mem: torch.Tensor, ego, pool: torch.Tensor, parents: torch.Tensor,
                       pool_out: Dict, k: Optional[int] = None, valid: Optional[torch.Tensor] = None,
                       sel_score: Optional[torch.Tensor] = None, pref: Optional[torch.Tensor] = None,
                       sample: bool = False, generator: Optional[torch.Generator] = None) -> Dict:
        """refiner v4_3 M5 computed edits (ck/v4/edits.py; V4Cfg.edits 'computed', readout 'range'): the top-k
        (edit_k) PARENT columns of pool [T, N, 8, 3] (parents [T, N, 8, 3]; the columns equal to their parent, one per
        distinct parent, edits.parent_columns) by sel_score [T, N] (the deployment selection score) or the critic-only
        selection score of pool_out['score_logit'] (invalid columns excluded), each edited by the 0-parameter rule on
        the PREDICTED fields of pool_out = score_set(..., return_pctx=True) of the pool (bisection with the LQR
        executed states; no edit -> the slot holds the source itself), re-scored on the source's parent corridor.
        Extra candidates only (never a gate).  -> the correct_topk keys with their shapes src_idx [T, k], edit_traj
        [T, k, 8, 3], edit_score_logit [T, k, 5], edit_w_lat [T, k, 6], edit_haz, u [T, k, 12] (the corrector's
        controls on edit_base; 0 without the corrector), + edit_base [T, k, 8, 3] (the computed edit), u_rule
        [T, k, 2] (a_lon m/s^2, d_lat m), edit_kind [T, k] (0 none / 1 lon / 2 lat / 3 both), edit_need (keys outside
        the predicted tube, same code), a_lon / d_lat / src_valid (False = a filler slot) / edit_clear / src_score
        [T, k]; with V4Cfg.corrector also its zero-init residual on the edit (mu / log_sig / logp / ctx; pool_out then
        needs return_ctx=True)."""
        if not self.v4.computed_edits:
            raise ValueError("computed_edits: this CK net has V4Cfg.edits != 'computed'")
        from .v4.edits import computed_edits
        return computed_edits(self, feat, mem, ego, pool, parents, pool_out, k=k, valid=valid, sel_score=sel_score,
                              pref=pref, sample=sample, generator=generator)

    def corrector_logp(self, feat: torch.Tensor, mem: torch.Tensor, ego, src: torch.Tensor,
                       src_par: Optional[torch.Tensor], u: torch.Tensor, pref: Optional[torch.Tensor] = None,
                       parent_index: Optional[torch.Tensor] = None, pctx: Optional[Dict] = None) -> torch.Tensor:
        """log pi(u | ctx(src)) [T, k] under the CURRENT corrector for recorded sources src [T, k, 8, 3] (parents
        src_par) and recorded controls u [T, k, 12] (AWR on official labels of earlier generations).
        parent_index [T, k] + pctx (score_set(..., return_pctx=True) of a set with these parents and the same scene):
        re-use that parent pass (src_par then None or pctx['parents']); without them the parents are deduped as in
        score_set."""
        with torch.no_grad():   # the critic only supplies the (detached) context
            o = self.score_set(feat, mem, ego, src, src_par, pref=pref, return_ctx=True, parent_index=parent_index,
                               pctx=pctx)
        T, k = src.shape[:2]
        ctx = torch.cat([o["ctx"], o["score_logit"].detach().float(),
                         (ego[0].float() / 15.0).reshape(T, 1, 1).expand(T, k, 1)], -1)
        a = self.corrector(ctx)
        return self.corrector.log_prob(a["mu"], a["log_sig"], u)

    # ------------------------------------------------------------------------------------------ forward
    def forward(self, bev: Optional[torch.Tensor], cand: torch.Tensor, status: torch.Tensor,
                extra: Optional[torch.Tensor] = None, decode: bool = True, slope: float = 0.0,
                rescore_corr: bool = False, objs: Optional[Dict[str, torch.Tensor]] = None,
                cand_parent: Optional[torch.Tensor] = None, extra_parent: Optional[torch.Tensor] = None,
                priv: Optional[Dict[str, torch.Tensor]] = None, rel_objs: Optional[Dict[str, torch.Tensor]] = None,
                refine_cols: Optional[torch.Tensor] = None) -> Dict:
        """cand_parent / extra_parent [T, K(2), 8, 3]: refiner v4 hazard readout only (each candidate's parent draft;
        None = itself); ignored by the mean readout.  priv: the ref_obj_* dict of a PRIVILEGED range teacher
        (score_set)."""
        if self.v4_readout is not None:
            return self._forward_v4(bev, cand, status, extra, decode, slope, rescore_corr, objs, cand_parent,
                                    extra_parent, priv, rel_objs, refine_cols)
        if priv is not None:
            raise ValueError("priv given to a CK net without a range readout")
        if rel_objs is not None or refine_cols is not None:
            raise ValueError("rel_objs / refine_cols given to a CK net without the v5 modules")
        T, K = cand.shape[:2]
        if status.shape[0] != T:
            raise ValueError(f"status {tuple(status.shape)} vs cand {tuple(cand.shape)}")
        if self.temporal is None and objs is not None:
            raise ValueError("objs given to a CK net without a temporal branch")
        if self.temporal is not None and objs is None:
            raise ValueError("this CK net has a temporal branch: forward(..., objs=...) is required")
        ego = ego_inputs(status)
        v0, a0, eds, cmd = ego
        cand = cand.float()
        feat, mem = self.trunk.scene(bev, T)
        ctx = self.temporal.scene(objs) if self.temporal is not None else None
        tf = (lambda Kc: None) if ctx is None else (lambda Kc: self.temporal.hook(ctx, T, Kc))
        o = self.trunk.candidates(feat, mem, cand, v0, a0, eds, cmd, tfeat=tf(K))
        out = {"score_logit": self.score(o, T, K), "z_lon": o["z_lon"].float(), "w_lat": o["w_lat"].float(),
               "gate_logit": o["gate_logit"].float(), "ego": ego}
        if decode or rescore_corr:
            out["corr"] = correct(cand, out["z_lon"], out["w_lat"], v0, slope)
        if rescore_corr:
            ct = out["corr"]["traj"].detach()
            out["corr_score_logit"] = self.score(self.trunk.candidates(feat, mem, ct, v0, a0, eds, cmd, tfeat=tf(K)),
                                                 T, K)
        if extra is not None:
            if extra.shape[0] != T:
                raise ValueError(f"extra {tuple(extra.shape)} vs cand {tuple(cand.shape)}")
            K2 = extra.shape[1]
            out["extra_score_logit"] = self.score(
                self.trunk.candidates(feat, mem, extra.float(), v0, a0, eds, cmd, tfeat=tf(K2)), T, K2)
        if self.lead_head is not None:
            out["lead_logit"] = self.lead_head(o["h"].reshape(T, K, -1).mean(1)).float()[:, 0]
        if ctx is not None:
            if ctx.fc_raw is not None:
                out["fc"] = ctx.fc_raw                                      # learned forecast (forecast loss)
            out["t_stats"] = {"n_obj": ctx.n_obj, "n_valid": ctx.n_valid}
        if self.pref_head is not None:
            out["pref_logp"] = self.scene_pref(mem, ego)
        return out

    def _forward_v4(self, bev, cand, status, extra, decode, slope, rescore_corr, objs, cand_parent,
                    extra_parent, priv=None, rel_objs=None, refine_cols=None) -> Dict:
        """forward() with the v4 hazard readout (same output keys; + 'haz' (per set: a_nc / a_ttc / a_dac / haz_mask /
        t_arr / d_lat) and 'pref_logp' [T] when the P_ref head is on)."""
        T, K = cand.shape[:2]
        if status.shape[0] != T:
            raise ValueError(f"status {tuple(status.shape)} vs cand {tuple(cand.shape)}")
        if objs is not None:
            raise ValueError("objs given to a CK net without a temporal branch")
        ego = ego_inputs(status)
        v0 = ego[0]
        cand = cand.float()
        feat, mem = self.trunk.scene(bev, T)
        pref = self.scene_pref(mem, ego)
        refine = refine_cols is not None
        if refine and self.corrector_v5 is None:
            raise ValueError("refine_cols given to a CK net without the v5 corrector")
        o = self.score_set(feat, mem, ego, cand, cand_parent, pref=pref, priv=priv, rel_objs=rel_objs,
                           return_ctx=refine)
        out = {"score_logit": o["score_logit"], "z_lon": o["z_lon"], "w_lat": o["w_lat"],
               "gate_logit": o["gate_logit"], "ego": ego, "haz": o["haz"]}
        if refine:                                       # refiner v5 corrector A on the given draft columns
            out["refine"] = self.refine_drafts(o, cand, refine_cols, ego)
        if pref is not None:
            out["pref_logp"] = pref
        if decode or rescore_corr:
            out["corr"] = correct(cand, out["z_lon"], out["w_lat"], v0, slope)
        if rescore_corr:
            ct = out["corr"]["traj"].detach()
            out["corr_score_logit"] = self.score_set(feat, mem, ego, ct, cand, pref=pref, priv=priv,
                                                     rel_objs=rel_objs)["score_logit"]
        if extra is not None:
            if extra.shape[0] != T:
                raise ValueError(f"extra {tuple(extra.shape)} vs cand {tuple(cand.shape)}")
            oe = self.score_set(feat, mem, ego, extra.float(), extra_parent, pref=pref, priv=priv, rel_objs=rel_objs)
            out["extra_score_logit"] = oe["score_logit"]
            out["extra_haz"] = oe["haz"]
        return out


    def refine_drafts(self, o: Dict, cand: torch.Tensor, cols: torch.Tensor, ego) -> Dict:
        """refiner v5 corrector A (ck/v5/corrector.py) on the draft columns cols [T, R] of a score_set output o (with
        return_ctx): mu from the critic context (detached), the per-state reads zq (NOT detached) and the top related
        objects; tau1 = decode(tau0, mu_lon, w_lat_head (detached) + mu_lat), decoder mode B, differentiable in mu.
        -> {cols, tau0 [T, R, 8, 3] (the drafts), tau_base (mu = 0: the laterally corrected drafts, no grad), tau1,
        mu / log_sig [T, R, 12]}."""
        from .v5.corrector import decode_grad
        a, src, wl, cols = self._v5_act(o, cand, cols, ego)
        tau1 = decode_grad(src, a["mu"][..., :N_FREE], wl + a["mu"][..., N_FREE:], ego[0])
        with torch.no_grad():               # mu = 0: today's laterally corrected draft (the corrector's own baseline)
            tau_base = decode_grad(src, torch.zeros_like(wl), wl, ego[0])
        return {"cols": cols, "tau0": src, "tau_base": tau_base, "tau1": tau1, "mu": a["mu"], "log_sig": a["log_sig"]}

    def _v5_act(self, o: Dict, cand: torch.Tensor, cols: torch.Tensor, ego):
        """the v5 corrector's policy on the draft columns cols [T, R] of a score_set output o (return_ctx) -> (a =
        {mu, log_sig} [T, R, 12], drafts [T, R, 8, 3], lateral head w_lat (detached) [T, R, 6], cols)."""
        T, R = cols.shape
        cols = cols.to(cand.device).long()
        g4 = lambda a: torch.gather(a, 1, cols.reshape(T, R, *([1] * (a.dim() - 2))).expand(T, R, *a.shape[2:]))
        haz = o["haz"]
        zs = int(self.v5.z_stride)
        from .v4 import lqr as _LQR
        tz = (torch.arange(_LQR.NT, dtype=torch.float32, device=cand.device) * _LQR.DT)[::zs]
        zq = g4(haz["zq"])
        if getattr(self, "z_adapter", None) is not None:   # V5Cfg.z_adapt: P_det / P_map on teacher-normalised Z
            zq = self.z_adapter(zq)
        zc = getattr(self.v5, "z_channels", "all")
        if zc != "all":                     # per-teacher channel mask (V5Cfg.z_channels): zeroed, no gradient
            keep = torch.zeros(zq.shape[-1], dtype=zq.dtype, device=zq.device)
            idx = [0, 1, 3] + list(range(4, zq.shape[-1])) if zc == "det" else [2, 3]
            keep[idx] = 1.0
            zq = zq * keep
        top = g4(haz["rel_top"]) if "rel_top" in haz else None
        if top is not None and self.v5.top_feats == "v2":   # + Delta r of the probes (design §3.7 A_S input)
            top = self._v5_top_v2(haz, g4, g4(cand.float()), ego, top)
        a = self.corrector_v5(g4(o["ctx"]), g4(o["score_logit"].detach().float()), ego[0], zq, tz, top)
        return a, g4(cand.float()), g4(o["w_lat"]).detach().float(), cols

    def _v5_probes(self, src: torch.Tensor, ego) -> Tuple[torch.Tensor, torch.Tensor]:
        """top_feats 'v2' Delta r probes (design §3.7): the CK2 variant rule's a-0.5, l-0.5, l+0.5 of each draft src
        [T, R, 8, 3] (ck/variants.make_variants 'separate' with the run's s_on_frac / compute dtype = V5Cfg.probe_*, ego
        speed v0 = hypot(vx, vy) in float64 as cands2.v0_from_status: a parent draft's probes are its pool siblings
        bit for bit (columns c + 2 / c + 4 / c + 5); for a variant draft (a deployed pick may be one) they are variants
        of the variant -- a necessary extension of the design's parent probes, declared).  All three are mode-A
        variants, so variants.ext (mode B only) does not enter.  -> probes [T, R, 3, 8, 3] f32, ok [T, R, 3] (False: a
        lateral probe of a draft with S_8 < 3 m -- a duplicate of the draft -- or a non-finite draft)."""
        from .variants import make_variants
        x = src.detach().float()
        fin = torch.isfinite(x).flatten(2).all(-1)
        x = torch.where(fin[..., None, None], x, torch.zeros_like(x))
        eds = ego[2].detach()
        v0 = torch.hypot(eds[:, 0].double(), eds[:, 1].double())
        dt = torch.float64 if self.v5.probe_dtype == "float64" else torch.float32
        out = make_variants(x, v0, PROBE_SPEEDS, PROBE_LATS, "separate", float(self.v5.probe_s_on_frac),
                            compute_dtype=dt, ext="straight")
        return out["traj"][:, :, 1:], out["valid"][:, :, 1:] & fin[..., None]

    def _v5_top_v2(self, haz: Dict, g4, src: torch.Tensor, ego, top: torch.Tensor) -> torch.Tensor:
        """top_feats 'v2' corrector input [T, R, 3 * 13]: per top object the relation's 10 features (haz rel_top of the
        drafts src [T, R, 8, 3], g4 = the drafts' column gather) + Delta r_i(p) = a_i r_i(draft) - a_i r_i(probe p) for
        the probes a-0.5 / l-0.5 / l+0.5 (_v5_probes; r = max(r_nc, r_ttc), a r clipped as the relation's top score),
        re-read by relation_reread with the SAME token rows (haz rel_objs_tok); 0 for an empty top slot (top_idx -1) or
        an invalid probe.  No graph through Delta r (an explanation input, like the detached critic context; the a r
        entries of the 10 keep theirs).  A deterministic function of (draft, ego, rows): the recorded policy, its AWR
        replay and the inference see identical inputs."""
        from .v5.config import N_TOP
        from .v5.relation import N_TOP_REL_V2, P_MAX
        T, R = src.shape[:2]
        with torch.no_grad(), torch.autocast(device_type=src.device.type, enabled=False):
            probes, pok = self._v5_probes(src, ego)
            rr = self.relation_reread(haz["rel_objs_tok"], probes, ego)
            a = haz["rel_a"].detach().float()                                                  # [T, N]
            ar = lambda w, r_nc, r_tt: torch.maximum((w * r_nc.float()).clamp(max=P_MAX),       # noqa: E731
                                                     (w * r_tt.float()).clamp(max=P_MAX))
            ar_p = ar(a[:, None, None], rr["r_nc"], rr["r_ttc"])                                # [T, R, 3p, N]
            ar_0 = ar(a[:, None], g4(haz["rel_r_nc"]), g4(haz["rel_r_ttc"]))                    # [T, R, N]
            ti = g4(haz["rel_top_idx"]).long()                                                  # [T, R, 3o]
            tic = ti.clamp(min=0)
            a0t = torch.gather(ar_0, 2, tic)                                                    # [T, R, 3o]
            apt = torch.gather(ar_p, 3, tic[:, :, None, :].expand(T, R, ar_p.shape[2], tic.shape[-1]))  # [T,R,3p,3o]
            dr = (a0t[:, :, None, :] - apt).transpose(2, 3)                                     # [T, R, 3o, 3p]
            dr = torch.where((ti >= 0)[..., None] & pok[:, :, None, :], dr, torch.zeros_like(dr))
        return torch.cat([top.float().reshape(T, R, N_TOP, N_TOP_REL_V2), dr], -1).reshape(T, R, -1)

    def relation_reread(self, O: Dict[str, torch.Tensor], traj: torch.Tensor, ego, grad: bool = False) -> Dict:
        """refiner v5 relation RE-READ primitive (design §3.7: the Delta r probes of top_feats 'v2'; X1 = exec(tau1) for
        L_edit-Bernoulli, LQR / 'ref' by V4Cfg.exec_model): the relation channel of the token rows O (haz['rel_objs_tok'] of a score_set of the same T
        tokens) on arbitrary drafts traj [T, R, 8, 3] or [T, R, P, 8, 3], executed by the critic's own model
        (range_readout.candidate_states, V4Cfg.exec_model, ego (vx, ax) = eds[:, 0], eds[:, 2] as score_set) and read
        at the same strided states (V5Cfg.z_stride) -> {r_nc, r_ttc [T, R(, P), N], H [T, R(, P), Q', 2], a_tok [T, N],
        states [T R (P), Q', 6], tq [Q'] (s)}; equals score_set's haz rel_r_* / zq H of the same drafts.  grad False
        (default): no graph; True: the reader's graph (eps, cv_prior, reliability) and that of the rows / traj as given
        (the caller detaches what it must not train)."""
        if self.relation is None:
            raise ValueError("relation_reread: this net has no v5 relation channel")
        from .v4.range_readout import candidate_states
        lead = tuple(traj.shape[:-2])
        T = int(lead[0])
        per = int(np.prod(lead[1:])) if len(lead) > 1 else 1
        flat = traj.reshape(T * per, *traj.shape[-2:]).float()
        eds = ego[2].float()
        dev = flat.device
        zs = int(self.v5.z_stride)
        with torch.set_grad_enabled(bool(grad) and torch.is_grad_enabled()), \
                torch.autocast(device_type=dev.type, enabled=False):
            states, tq = candidate_states(flat, eds[:, 0].repeat_interleave(per), eds[:, 2].repeat_interleave(per),
                                          self.v4.exec_model)
            tok = torch.arange(T, device=dev).repeat_interleave(per)
            rel = self.relation.read(O, states[:, ::zs], tq[::zs], tok)
        sh = lambda a: a.reshape(*lead, *a.shape[1:])                                           # noqa: E731
        out = {"r_nc": sh(rel["r_nc"]), "r_ttc": sh(rel["r_ttc"]), "H": sh(rel["H"]), "a_tok": rel["a_tok"],
               "states": states[:, ::zs], "tq": tq[::zs]}
        if "lr_nc" in rel:                               # relation 'student': log r (report 102 N1)
            out["lr_nc"], out["lr_ttc"] = sh(rel["lr_nc"]), sh(rel["lr_ttc"])
        return out

    def v5_logp(self, feat: torch.Tensor, mem: torch.Tensor, ego, src: torch.Tensor, src_par: Optional[torch.Tensor],
                u: torch.Tensor, pref: Optional[torch.Tensor] = None,
                rel_objs: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """refiner v5 Stage-2 AWR (report 97 §2-7 (1), §4 L_AWR(tau1); report 99 N4): log pi(u | src) [T, k] under the
        CURRENT corrector A_S for recorded sources src [T, k, 8, 3] (parents src_par) and recorded controls u
        [T, k, 12].  The critic / readers only supply the (detached) context, zq and the related objects (as the v4
        corrector_logp): the AWR gradient trains the corrector, L_refine / L_H / Z-KD train the readers."""
        if self.corrector_v5 is None:
            raise ValueError("v5_logp: this net has no v5 corrector")
        with torch.no_grad():
            o = self.score_set(feat, mem, ego, src.float(), src_par, pref=pref, return_ctx=True, rel_objs=rel_objs)
        T, k = src.shape[:2]
        cols = torch.arange(k, device=src.device)[None].expand(T, k)
        a, _, _, _ = self._v5_act(o, src.float(), cols, ego)
        return self.corrector_v5.log_prob(a["mu"], a["log_sig"], u)


CKHead = CKNet          # task-text name


# ----------------------------------------------------------------------------------------------- build / io
def _sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def load_arm_norm(run_dir: Union[str, Path], arm: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(mean, std) of <run_dir>/norm.npz (arm T) or norm_map.npz (arm M); None for arms S / none."""
    if arm not in NORM_FILE:
        return None
    mean, std, _ = load_norm(Path(run_dir) / NORM_FILE[arm])
    return mean, std


def _ckpt_path(run_dir: Path, which: str) -> Path:
    p = Path(which)
    if p.suffix == ".pt" and p.is_file():
        return p
    return run_dir / f"ckpt_{which}.pt"


def _torch_load(p: Path) -> Dict:
    return torch.load(p, map_location="cpu", weights_only=False)     # local trusted run files


def build_ck(arm: str, seed: int = 0, norm: Optional[Tuple[np.ndarray, np.ndarray]] = None,
             init_from: Optional[Union[str, Path]] = None, lead_aux: bool = False, score_hidden: int = 256,
             temporal: Optional[TemporalCfg] = None, corridor_version: int = 1, v4: Optional[V4Cfg] = None,
             v5: Optional[V5Cfg] = None) -> CKNet:
    """CKNet, optionally initialised from
      * a v1 stage-T run dir (config.json + ckpt_best.pt (else ckpt_last.pt) with RefinerNet keys): loaded into
        net.trunk; the source arm must equal the target arm, except target 'S', for which every adapter.* key is
        skipped (trunk only).  Missing / unexpected trunk keys are an error.  For T / M without `norm` the source
        run's norm.npz / norm_map.npz is used (the checkpoint carries the same buffers anyway);
      * a CK run dir (ckpt_last.pt whose 'model' keys start with 'trunk.'): full CKNet state, strict.
    For T / M from a v1 or CK run the checkpoint's adapter z-score buffers take precedence over `norm`.
    corridor_version (refiner-fix B1) is the TARGET's corridor geometry; it is not part of the state dict, so a run
    may be initialised from a run of another corridor version (fine-tuning across the fix is allowed, not checked).
    net.init_report = {source, kind, ckpt, ckpt_sha16, source_arm, n_loaded, skipped, missing, unexpected}."""
    report: Dict = {"source": None}
    if init_from is not None:
        src = Path(init_from)
        cfg = json.loads((src / "config.json").read_text()) if (src / "config.json").is_file() else {}
        src_arm = cfg.get("arm")
        tsrc = TemporalCfg.from_dict(cfg.get("temporal")).normalised()
        if tsrc != (temporal if temporal is not None else TemporalCfg()).normalised():
            raise ValueError(f"init_from {src}: temporal cfg {tsrc.to_dict()} differs from the target's")
        if norm is None and arm in NORM_FILE and src_arm == arm and (src / NORM_FILE[arm]).is_file():
            norm = load_arm_norm(src, arm)
    net = CKNet(arm, seed, norm, score_hidden=score_hidden, lead_aux=lead_aux, temporal=temporal,
                corridor_version=corridor_version, v4=v4, **({"v5": v5} if v5 is not None and v5.on else {}))
    if init_from is None:
        net.init_report = report
        return net
    p = src / "ckpt_last.pt"
    sd = _torch_load(p)["model"] if p.is_file() else None
    if sd is not None and any(k.startswith("trunk.") for k in sd):
        if src_arm is not None and src_arm != arm:
            raise ValueError(f"init_from CK run {src} has arm {src_arm!r}, target arm {arm!r}")
        net.load_state_dict(sd, strict=True)
        report.update(kind="ck", ckpt=str(p), n_loaded=len(sd), skipped=[], missing=[], unexpected=[])
    else:
        p = src / "ckpt_best.pt"
        if not p.is_file():
            p = src / "ckpt_last.pt"
        sd = _torch_load(p)["model"]
        if src_arm is None:
            raise ValueError(f"init_from v1 run {src} has no config.json 'arm'")
        if arm == "S":
            skipped = sorted(k for k in sd if k.startswith("adapter."))
            sd = {k: v for k, v in sd.items() if not k.startswith("adapter.")}
        elif src_arm == arm:
            skipped = []
        else:
            raise ValueError(f"init_from v1 run {src}: source arm {src_arm!r} cannot initialise target arm {arm!r}")
        res = net.trunk.load_state_dict(sd, strict=False)
        missing = list(res.missing_keys)
        if arm == "S":
            missing = [k for k in missing if not k.startswith("adapter.")]
        if missing or res.unexpected_keys:
            raise ValueError(f"init_from {src}: missing {missing} unexpected {list(res.unexpected_keys)}")
        report.update(kind="v1", ckpt=str(p), n_loaded=len(sd), skipped=skipped, missing=missing,
                      unexpected=list(res.unexpected_keys))
    report.update(source=str(src), source_arm=src_arm, ckpt_sha16=_sha16(p))
    net.init_report = report
    return net


def init_partial(net: CKNet, run_dir: Union[str, Path], which: str = "last") -> Dict:
    """refiner v5 warm start from a trained CK / v4_3 run (e.g. the range teacher v4T10_rng2): every parameter / buffer
    of the source whose name and shape match is copied; the range readout's residual first layer (more inputs in v5:
    [ev(7), g(d), extra]) gets the source columns and ZERO new columns, so the warm-started critic's outputs equal the
    source's at step 0 (the v5 relation term starts at rel_w = 0).  The source arm must equal the target's.  ->
    report {source, ckpt, ckpt_sha16, n_copied, adapted, new (target keys not in the source), skipped (shape)}."""
    src = Path(run_dir)
    cfg = json.loads((src / "config.json").read_text()) if (src / "config.json").is_file() else {}
    if cfg.get("arm") not in (None, net.arm):
        raise ValueError(f"init_partial: source {src} arm {cfg.get('arm')!r} != target {net.arm!r}")
    p = _ckpt_path(src, which)
    sd = _torch_load(p)["model"]
    own = net.state_dict()
    copied, adapted, skipped = 0, [], []
    with torch.no_grad():
        for k, v in sd.items():
            if k not in own:
                skipped.append(k)
                continue
            t = own[k]
            if tuple(t.shape) == tuple(v.shape):
                t.copy_(v.to(t.dtype))
                copied += 1
            elif k == "v4_readout.resid.0.weight" and t.dim() == 2 and t.shape[0] == v.shape[0] \
                    and t.shape[1] > v.shape[1]:
                t.zero_()
                t[:, :v.shape[1]].copy_(v.to(t.dtype))
                adapted.append(k)
            else:
                skipped.append(k)
    new = sorted(set(own) - set(sd))
    return {"source": str(src), "ckpt": str(p), "ckpt_sha16": _sha16(p), "n_copied": copied, "adapted": adapted,
            "new": new, "skipped": skipped}


def save_ck(path: Union[str, Path], net: CKNet, cfg: Dict, **extra) -> None:
    """torch.save({'model', 'cfg', **extra}) atomically (tmp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    torch.save({"model": net.state_dict(), "cfg": cfg, **extra}, tmp)
    os.replace(tmp, path)


def load_ck(run_dir: Union[str, Path], which: str = "last", device="cpu",
            allow_privileged: bool = False) -> Tuple[CKNet, Dict]:
    """Rebuild a CK run (config.json: arm, seed, optional lead_aux / score_hidden / temporal / corridor_version
    (absent = 1, every run before refiner-fix); norm.npz (T) /
    norm_map.npz (M)) and load ckpt_<which>.pt['model'] strictly (which = 'last', 'ep<e>' or a .pt path) ->
    (net.eval(), cfg).  A PRIVILEGED run (temporal oracle / gt source) raises PermissionError unless
    allow_privileged (single choke point for KD teachers, navtest / e2e evals, fit_kd_calib)."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    arm = cfg["arm"]
    tcfg = TemporalCfg.from_dict(cfg.get("temporal"))
    if tcfg.privileged and not allow_privileged:
        raise PermissionError(f"CK run {run_dir} is PRIVILEGED (temporal forecaster {tcfg.forecaster!r}, source "
                              f"{tcfg.source!r}): analysis only, pass allow_privileged=True")
    rg_priv = str((cfg.get("v4") or {}).get("rg_priv", "off"))
    if rg_priv != "off" and not allow_privileged:
        raise PermissionError(f"CK run {run_dir} is PRIVILEGED (v4.rg_priv {rg_priv!r}: GT t0 objects as input): "
                              f"analysis only, pass allow_privileged=True")
    # the checkpoint carries the adapter's z-score buffers, so a missing norm file is not fatal
    norm = load_arm_norm(run_dir, arm) if arm in NORM_FILE and (run_dir / NORM_FILE[arm]).is_file() else None
    net = CKNet(arm, int(cfg.get("seed", 0)), norm,
                score_hidden=int(cfg.get("score_hidden", 256)), lead_aux=bool(int(cfg.get("lead_aux", 0) or 0)),
                temporal=tcfg, ca_b=(torch.ones(2, 10) if tcfg.on and tcfg.forecaster != "learned" else None),
                corridor_version=1 if cfg.get("corridor_version") is None else int(cfg["corridor_version"]),
                v4=V4Cfg.from_dict(cfg.get("v4")), v5=V5Cfg.from_dict(cfg.get("v5")))
    # (ca_b placeholder: the checkpoint's persistent buffer overwrites it in the strict load below)
    p = _ckpt_path(run_dir, which)
    net.load_state_dict(_torch_load(p)["model"], strict=True)
    net.init_report = {"source": str(run_dir), "kind": "ck", "ckpt": str(p), "ckpt_sha16": _sha16(p)}
    return net.to(device).eval(), cfg
