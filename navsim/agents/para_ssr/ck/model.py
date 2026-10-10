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
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ..refiner.adapters import BEV_H, BEV_W, load_norm
from ..refiner.corridor import N_ST, corridor_geometry, corridor_tensor, global_tokens
from ..refiner.decoder import N_FREE
from ..refiner.e2e import AdapterS, ego_inputs
from ..refiner.geometry import N_POSE
from ..refiner.refiner_net import ADAPTER_SEED_OFFSET, RefinerNet, draft_features
from .constants import CK_KEYS, NORM_FILE, SCORE_SEED_OFFSET
from .correct import correct

ARMS = ("T", "M", "S", "none")
N_KEYS = len(CK_KEYS)


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
                   a0: torch.Tensor, eds: torch.Tensor, cmd: torch.Tensor) -> Dict[str, torch.Tensor]:
        """tau [T, K, 8, 3] -> h [T*K, 2d] (not detached), z_lon / w_lat [T, K, 6], gate_logit [T, K], ud, geom."""
        T, K = tau.shape[:2]
        B = T * K
        with torch.autocast(device_type=tau.device.type, enabled=False):
            flat = tau.reshape(B, N_POSE, 3).float()
            geom = corridor_geometry(flat)
            rep = lambda x: x.reshape(T, *x.shape[1:]).repeat_interleave(K, 0)
            ud = draft_features(flat, rep(v0.float()), rep(a0.float()), rep(eds.float()), rep(cmd), geom)
        X = corridor_tensor(feat, geom, T, K)                                         # [B, 70, 48, 17]
        Hc = self.enc(X)                                                              # [B, 128, 48, 17]
        st = self.station(Hc.permute(0, 2, 1, 3).reshape(B, N_ST, -1))               # [B, 48, d]
        x = torch.cat([self.draft_mlp(ud.to(st.dtype))[:, None], st], 1)              # [B, 49, d]
        for layer in self.layers:
            x = layer(x, mem, T, K)
        x = self.norm_out(x)
        xd = x[:, 0]
        gate = self.gate_head(xd.detach())[:, 0]
        h = torch.cat([xd, x[:, 1:].mean(1)], -1)
        return {"h": h, "gate_logit": gate.reshape(T, K), "z_lon": self.lon_head(h).reshape(T, K, N_FREE),
                "w_lat": self.lat_head(h).reshape(T, K, N_FREE), "ud": ud, "geom": geom}

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
                 score_hidden: int = 256, lead_aux: bool = False):
        super().__init__()
        self.arm = arm
        self.seed = int(seed)
        self.lead_aux = bool(lead_aux)
        self.trunk = build_trunk(arm, seed, norm)
        d2 = 2 * self.trunk.d_model
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + SCORE_SEED_OFFSET)
            self.score_head = nn.Sequential(nn.Linear(d2, score_hidden), nn.GELU(), nn.Linear(score_hidden, N_KEYS))
            self.lead_head = (nn.Sequential(nn.Linear(d2, 128), nn.GELU(), nn.Linear(128, 1)) if lead_aux else None)
        self.init_report: Dict = {}

    # ------------------------------------------------------------------------------------------ helpers
    def param_counts(self) -> Dict[str, int]:
        c = self.trunk.param_counts()
        c["score_head"] = sum(p.numel() for p in self.score_head.parameters())
        c["lead_head"] = 0 if self.lead_head is None else sum(p.numel() for p in self.lead_head.parameters())
        c["total"] = c["total"] + c["score_head"] + c["lead_head"]
        return c

    @torch.no_grad()
    def set_score_prior(self, prior: Sequence[float]) -> None:
        """Set the score head's output bias to logit(prior) per key (CK_KEYS order), e.g. label means."""
        p = torch.as_tensor(np.asarray(prior, np.float64)).clamp(1e-4, 1 - 1e-4)
        self.score_head[-1].bias.copy_(torch.log(p / (1 - p)).to(self.score_head[-1].bias))

    def score(self, cand_out: Dict[str, torch.Tensor], T: int, K: int) -> torch.Tensor:
        return self.score_head(cand_out["h"]).float().reshape(T, K, N_KEYS)

    # ------------------------------------------------------------------------------------------ forward
    def forward(self, bev: Optional[torch.Tensor], cand: torch.Tensor, status: torch.Tensor,
                extra: Optional[torch.Tensor] = None, decode: bool = True, slope: float = 0.0,
                rescore_corr: bool = False) -> Dict:
        T, K = cand.shape[:2]
        if status.shape[0] != T:
            raise ValueError(f"status {tuple(status.shape)} vs cand {tuple(cand.shape)}")
        ego = ego_inputs(status)
        v0, a0, eds, cmd = ego
        cand = cand.float()
        feat, mem = self.trunk.scene(bev, T)
        o = self.trunk.candidates(feat, mem, cand, v0, a0, eds, cmd)
        out = {"score_logit": self.score(o, T, K), "z_lon": o["z_lon"].float(), "w_lat": o["w_lat"].float(),
               "gate_logit": o["gate_logit"].float(), "ego": ego}
        if decode or rescore_corr:
            out["corr"] = correct(cand, out["z_lon"], out["w_lat"], v0, slope)
        if rescore_corr:
            ct = out["corr"]["traj"].detach()
            out["corr_score_logit"] = self.score(self.trunk.candidates(feat, mem, ct, v0, a0, eds, cmd), T, K)
        if extra is not None:
            if extra.shape[0] != T:
                raise ValueError(f"extra {tuple(extra.shape)} vs cand {tuple(cand.shape)}")
            K2 = extra.shape[1]
            out["extra_score_logit"] = self.score(
                self.trunk.candidates(feat, mem, extra.float(), v0, a0, eds, cmd), T, K2)
        if self.lead_head is not None:
            out["lead_logit"] = self.lead_head(o["h"].reshape(T, K, -1).mean(1)).float()[:, 0]
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
             init_from: Optional[Union[str, Path]] = None, lead_aux: bool = False, score_hidden: int = 256) -> CKNet:
    """CKNet, optionally initialised from
      * a v1 stage-T run dir (config.json + ckpt_best.pt (else ckpt_last.pt) with RefinerNet keys): loaded into
        net.trunk; the source arm must equal the target arm, except target 'S', for which every adapter.* key is
        skipped (trunk only).  Missing / unexpected trunk keys are an error.  For T / M without `norm` the source
        run's norm.npz / norm_map.npz is used (the checkpoint carries the same buffers anyway);
      * a CK run dir (ckpt_last.pt whose 'model' keys start with 'trunk.'): full CKNet state, strict.
    For T / M from a v1 or CK run the checkpoint's adapter z-score buffers take precedence over `norm`.
    net.init_report = {source, kind, ckpt, ckpt_sha16, source_arm, n_loaded, skipped, missing, unexpected}."""
    report: Dict = {"source": None}
    if init_from is not None:
        src = Path(init_from)
        cfg = json.loads((src / "config.json").read_text()) if (src / "config.json").is_file() else {}
        src_arm = cfg.get("arm")
        if norm is None and arm in NORM_FILE and src_arm == arm and (src / NORM_FILE[arm]).is_file():
            norm = load_arm_norm(src, arm)
    net = CKNet(arm, seed, norm, score_hidden=score_hidden, lead_aux=lead_aux)
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


def load_ck(run_dir: Union[str, Path], which: str = "last", device="cpu") -> Tuple[CKNet, Dict]:
    """Rebuild a CK run (config.json: arm, seed, optional lead_aux / score_hidden; norm.npz (T) / norm_map.npz (M))
    and load ckpt_<which>.pt['model'] strictly (which = 'last', 'ep<e>' or a .pt path) -> (net.eval(), cfg)."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    arm = cfg["arm"]
    # the checkpoint carries the adapter's z-score buffers, so a missing norm file is not fatal
    norm = load_arm_norm(run_dir, arm) if arm in NORM_FILE and (run_dir / NORM_FILE[arm]).is_file() else None
    net = CKNet(arm, int(cfg.get("seed", 0)), norm,
                score_hidden=int(cfg.get("score_hidden", 256)), lead_aux=bool(int(cfg.get("lead_aux", 0) or 0)))
    p = _ckpt_path(run_dir, which)
    net.load_state_dict(_torch_load(p)["model"], strict=True)
    net.init_report = {"source": str(run_dir), "kind": "ck", "ckpt": str(p), "ckpt_sha16": _sha16(p)}
    return net.to(device).eval(), cfg
