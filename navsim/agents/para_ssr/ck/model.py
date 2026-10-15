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

ARMS = ("T", "M", "S", "none")
N_KEYS = len(CK_KEYS)
V4_SEED_OFFSET = 40414      # refiner v4 heads: own seeded RNG, built after every CK2 / v3 module
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
        Same modules as candidates() (no heads)."""
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
                 ca_b: Optional[torch.Tensor] = None, corridor_version: int = 1, v4: Optional[V4Cfg] = None):
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
                          ("corrector", self.corrector)):
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

    def score_set(self, feat: torch.Tensor, mem: torch.Tensor, ego, cand: torch.Tensor,
                  parents: Optional[torch.Tensor] = None, tfeat=None, pref: Optional[torch.Tensor] = None,
                  return_ctx: bool = False) -> Dict:
        """Score one candidate set with the token's scene (feat, mem from trunk.scene):
        cand [T, K, 8, 3] -> score_logit [T, K, 5] f32, w_lat / z_lon [T, K, 6] f32, gate_logit [T, K], and for the v4
        parent-pass readouts 'haz' = {t_arr (bin-centre first arrival), t_in / t_out (entry / min(leave, horizon) of
        the station bins), d_lat, haz_mask [T, K, 48], is_self [T, K]; hazard: a_nc / a_ttc / a_dac
        [T, K, 48]; evidence: occ_logit [T, K, 48, 17, M], margin [T, K, 48, 17], pts [T, K, 48, 17, 2] (parent grid
        points, N frame), lp_nc / lp_ttc / lp_dac / min_margin [T, K]}.
        readout 'mean': exactly trunk.candidates + score (the CK2 / v3 ops; parents ignored).
        readout 'hazard': parents [T, K, 8, 3] = each candidate's parent draft (None: every candidate is its own
        parent); the scene is read on the parent's corridor, the candidate reads it at its own schedule."""
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
        par = cand if parents is None else parents.float()
        if tuple(par.shape) != tuple(cand.shape):
            raise ValueError(f"score_set: parents {tuple(par.shape)} vs cand {tuple(cand.shape)}")
        P = self.trunk.stations(feat, mem, par, v0, a0, eds, cmd, time_free=self.v4.time_free)
        flat = cand.reshape(B, N_POSE, 3)
        rep = lambda x: x.reshape(T, *x.shape[1:]).repeat_interleave(K, 0)
        with torch.autocast(device_type=cand.device.type, enabled=False):
            is_self = (flat == par.reshape(B, N_POSE, 3)).flatten(1).all(1)
            sib = on_parent(P["geom"], flat, is_self)
            cgeom = P["geom"] if parents is None else corridor_geometry(flat, version=self.corridor_version)
            ud = P["ud"] if parents is None else draft_features(flat, rep(v0.float()), rep(a0.float()),
                                                               rep(eds.float()), rep(cmd), cgeom)
        x = P["x"]
        g = self.trunk.draft_mlp(ud.to(x.dtype))
        pr = None if pref is None else rep(pref.reshape(T, 1).float())[:, 0]
        sh = lambda a: a.reshape(T, K, *a.shape[1:])
        if self.v4.hazard:
            e = feasibility(cgeom.S, rep(v0.float().reshape(T, 1))[:, 0])
            r = self.v4_readout(x[:, 1:], x[:, 0], P["Hc"], g, sib["t_in"], sib["t_leave"], sib["v"], sib["d"],
                                cgeom.S[:, -1], e, pr)
            haz = {k: sh(r[k]) for k in ("a_nc", "a_ttc", "a_dac")}
        else:
            r = self.v4_readout(x[:, 1:], x[:, 0], P["Hc"], g, P["geom"].s, sib["s_knots"], sib["d_knots"], sib["d"],
                                sib["t_in"], sib["t_leave"], cgeom.S[:, -1], pr)
            haz = {k: sh(r[k]) for k in ("occ_logit", "margin", "lp_nc", "lp_ttc", "lp_dac", "min_margin")}
            haz["pts"] = sh(P["geom"].points.detach().float())
        haz.update(haz_mask=sh(r["mask"]), t_arr=sh(sib["t"]), t_in=sh(sib["t_in"]), t_out=sh(r["t_out"]),
                   d_lat=sh(sib["d"]), is_self=sh(is_self))
        out = {"score_logit": sh(r["score_logit"]), "w_lat": sh(r["w_lat"]),
               "z_lon": torch.zeros(T, K, N_FREE, dtype=torch.float32, device=cand.device),
               "gate_logit": torch.zeros(T, K, dtype=torch.float32, device=cand.device), "haz": haz}
        if return_ctx:          # corrector context (detached: the corrector never moves the critic's features)
            mf = r["mask"].to(x.dtype)[..., None]
            xs_mean = (x[:, 1:] * mf).sum(1) / mf.sum(1).clamp(min=1.0)
            out["ctx"] = sh(torch.cat([g, x[:, 0], xs_mean], -1).detach().float())
        return out

    def correct_topk(self, feat: torch.Tensor, mem: torch.Tensor, ego, pool: torch.Tensor, parents: torch.Tensor,
                     pool_out: Dict, pref: Optional[torch.Tensor] = None, sample: bool = False,
                     k: Optional[int] = None, valid: Optional[torch.Tensor] = None,
                     generator: Optional[torch.Generator] = None) -> Dict:
        """refiner v4 corrector A: shortlist the top-k of pool [T, N, 8, 3] by the critic selection score
        (corrector.critic_select_score of pool_out['score_logit'], invalid columns excluded), edit them (mu, or a
        sample when training), decode (mode B) and re-score the edits on their source's parent corridor.  pool_out
        = score_set(..., return_ctx=True) of the pool.  -> src_idx [T, k], u / mu / log_sig [T, k, 12], logp [T, k],
        edit_traj [T, k, 8, 3], edit_score_logit [T, k, 5], edit_w_lat [T, k, 6], edit_haz."""
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
        oe = self.score_set(feat, mem, ego, edits, src_par, pref=pref)
        return {"src_idx": idx, "u": a["u"], "mu": a["mu"], "log_sig": a["log_sig"], "logp": a["logp"],
                "edit_traj": edits, "edit_score_logit": oe["score_logit"], "edit_w_lat": oe["w_lat"],
                "edit_haz": oe.get("haz"), "ctx": ctx}

    def corrector_logp(self, feat: torch.Tensor, mem: torch.Tensor, ego, src: torch.Tensor, src_par: torch.Tensor,
                       u: torch.Tensor, pref: Optional[torch.Tensor] = None) -> torch.Tensor:
        """log pi(u | ctx(src)) [T, k] under the CURRENT corrector for recorded sources src [T, k, 8, 3] (parents
        src_par) and recorded controls u [T, k, 12] (AWR on official labels of earlier generations)."""
        with torch.no_grad():   # the critic only supplies the (detached) context
            o = self.score_set(feat, mem, ego, src, src_par, pref=pref, return_ctx=True)
        T, k = src.shape[:2]
        ctx = torch.cat([o["ctx"], o["score_logit"].detach().float(),
                         (ego[0].float() / 15.0).reshape(T, 1, 1).expand(T, k, 1)], -1)
        a = self.corrector(ctx)
        return self.corrector.log_prob(a["mu"], a["log_sig"], u)

    # ------------------------------------------------------------------------------------------ forward
    def forward(self, bev: Optional[torch.Tensor], cand: torch.Tensor, status: torch.Tensor,
                extra: Optional[torch.Tensor] = None, decode: bool = True, slope: float = 0.0,
                rescore_corr: bool = False, objs: Optional[Dict[str, torch.Tensor]] = None,
                cand_parent: Optional[torch.Tensor] = None, extra_parent: Optional[torch.Tensor] = None) -> Dict:
        """cand_parent / extra_parent [T, K(2), 8, 3]: refiner v4 hazard readout only (each candidate's parent draft;
        None = itself); ignored by the mean readout."""
        if self.v4_readout is not None:
            return self._forward_v4(bev, cand, status, extra, decode, slope, rescore_corr, objs, cand_parent,
                                    extra_parent)
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
                    extra_parent) -> Dict:
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
        o = self.score_set(feat, mem, ego, cand, cand_parent, pref=pref)
        out = {"score_logit": o["score_logit"], "z_lon": o["z_lon"], "w_lat": o["w_lat"],
               "gate_logit": o["gate_logit"], "ego": ego, "haz": o["haz"]}
        if pref is not None:
            out["pref_logp"] = pref
        if decode or rescore_corr:
            out["corr"] = correct(cand, out["z_lon"], out["w_lat"], v0, slope)
        if rescore_corr:
            ct = out["corr"]["traj"].detach()
            out["corr_score_logit"] = self.score_set(feat, mem, ego, ct, cand, pref=pref)["score_logit"]
        if extra is not None:
            if extra.shape[0] != T:
                raise ValueError(f"extra {tuple(extra.shape)} vs cand {tuple(cand.shape)}")
            oe = self.score_set(feat, mem, ego, extra.float(), extra_parent, pref=pref)
            out["extra_score_logit"] = oe["score_logit"]
            out["extra_haz"] = oe["haz"]
        return out


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
             temporal: Optional[TemporalCfg] = None, corridor_version: int = 1, v4: Optional[V4Cfg] = None) -> CKNet:
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
                corridor_version=corridor_version, v4=v4)
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
    # the checkpoint carries the adapter's z-score buffers, so a missing norm file is not fatal
    norm = load_arm_norm(run_dir, arm) if arm in NORM_FILE and (run_dir / NORM_FILE[arm]).is_file() else None
    net = CKNet(arm, int(cfg.get("seed", 0)), norm,
                score_hidden=int(cfg.get("score_hidden", 256)), lead_aux=bool(int(cfg.get("lead_aux", 0) or 0)),
                temporal=tcfg, ca_b=(torch.ones(2, 10) if tcfg.on and tcfg.forecaster != "learned" else None),
                corridor_version=1 if cfg.get("corridor_version") is None else int(cfg["corridor_version"]),
                v4=V4Cfg.from_dict(cfg.get("v4")))
    # (ca_b placeholder: the checkpoint's persistent buffer overwrites it in the strict load below)
    p = _ckpt_path(run_dir, which)
    net.load_state_dict(_torch_load(p)["model"], strict=True)
    net.init_report = {"source": str(run_dir), "kind": "ck", "ckpt": str(p), "ckpt_sha16": _sha16(p)}
    return net.to(device).eval(), cfg
