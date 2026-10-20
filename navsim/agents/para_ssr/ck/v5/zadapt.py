"""refiner v5 student Z adapters P_det / P_map with frozen teacher statistics (V5Cfg.z_adapt; design
refiner_v5_design_final.md §2.1 L82 / L86, §3.8 L191 / L212; Korean 97_refiner_v5_design.md §2-7).

Per-state reads Z of the student (zq = ck/v5/zread.zq_of + the relation hazard H; channel order Z_CHANNELS =
[b_nc, b_ttc, d_min, unknown, H_nc, H_ttc]) are first put in the TEACHERS' units, Z~ = (Z - mu_T) / sigma_T, with the
frozen per-channel statistics of the teacher readers on navtrain (buffers z_mu / z_sd, filled once from the
tools/ck/v5/fit_z_stats.py records of the DET teacher (b_nc, b_ttc, H_nc, H_ttc) and the MAP teacher (d_min); the
unknown flag passes bit for bit), then adapted per query j = 1..41 (every z_stride-th executed state):
  Z_S^det = P_det([b_nc, b_ttc, H_nc, H_ttc]~_j, unknown_j)   Z_S^map = P_map([d_min]~_j, unknown_j)
  P = per-channel affine (gamma init 1, beta 0) + one MLP (n + 1 -> hidden -> n, GELU, last layer zero-initialised),
so at init P(Z) == Z~ exactly.  Float32 with autocast off.  forward / normalise refuse (ValueError) until the
statistics are set (no silent identity).  CKNet._v5_act feeds the ADAPTED Z to the corrector A_S (training, the tau1
record, the AWR replay and the inference all pass there); the raw zq stays in score_set's haz (L_H reads its H channels
as probabilities); Z-KD (online2._v5_terms) compares the adapted student Z with the teacher's normalised Z (design §3.8
L212: SmoothL1(Z~_S, sg(Z~_T)), teacher-normalised units).  z_channels 'det' / 'map' (the corrector's per-teacher
channel mask) build only the kept half (no DDP-unused parameters).
Statistics records (STATS_VERSION; tools/ck/v5/fit_z_stats.py): per channel of TEACHER_CHANNELS[arm] the mean / std of
the teacher's zq over the mask stats_mask: a valid, finite pool column, a known state (unknown <= 0.5) and, for the B
channels, a READ state (zq > ZQ_FLOOR + 1e-6: the floor encodes 'not read', not a logit); std floored at STD_FLOOR
(and, when set into the adapter, the H channels at SD_FLOOR_CH = 0.25).
check_z_stats ties a record to its teacher run / checkpoint (run, which, arm, ckpt sha16, z_stride; 'unfinished'
records refused).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .zread import FLOOR

Tensor = torch.Tensor
STATS_VERSION = "v5_z_stats_v1"
Z_CHANNELS = ("b_nc", "b_ttc", "d_min", "unknown", "H_nc", "H_ttc")     # zq layout (zread.zq_of + relation H)
CH = {n: i for i, n in enumerate(Z_CHANNELS)}
UNK = CH["unknown"]
DET_CH = (CH["b_nc"], CH["b_ttc"], CH["H_nc"], CH["H_ttc"])            # P_det (H only with the relation channel)
MAP_CH = (CH["d_min"],)                                                # P_map
# the channels each teacher arm supervises (a DET teacher's D field and a MAP teacher's B / U fields are unsupervised:
# their statistics are never recorded, review 2026-10-20)
TEACHER_CHANNELS = {"T": ("b_nc", "b_ttc", "H_nc", "H_ttc"), "M": ("d_min",)}
TEACHER_OF_ARM = {"T": "det", "M": "map"}
B_CHANNELS = ("b_nc", "b_ttc")
ZQ_FLOOR = max(FLOOR / 5.0, -4.0)            # zread.zq_of: clip(FLOOR / 5, +-4) = a state that is not read
STD_FLOOR = 1e-3
# per-channel sigma floor applied when the statistics are SET (records keep the raw std): the hazard channels H are
# probabilities that are ~0 on most known states (DET teacher H_nc mean 0.0075, std 0.061 at 64 tokens), so the plain
# std puts H = 1 at ~16 sd and multiplies the Z-KD gradient on the raw H 55-270x against the raw-unit loss (workflow
# review 2026-10-20); 0.25 (the std of a Bernoulli(0.07)) keeps H = 1 at ~4 sd, the clip scale of the b channels
SD_FLOOR_CH = {"H_nc": 0.25, "H_ttc": 0.25}
N_MIN = 1000                                 # fewest states per channel of a usable record (fit_z_stats)
DRY_RUN_BELOW = 1024                         # a record of fewer tokens (or --limit) is a dry run (refused at launch)
MASK_RULE = ("valid finite pool column (online_variants valid96 & finite, bev_ok, obj_ok for a relation teacher) & "
             "unknown <= 0.5 & (b_nc / b_ttc only) zq > ZQ_FLOOR + 1e-6 (a read state); all strided states of the "
             "column (V5Cfg.z_stride)")


# ----------------------------------------------------------------------------------------------- statistics
def stats_mask(zq: Tensor, col_ok: Tensor, name: str) -> Tensor:
    """the states of channel `name` that enter the statistics: zq [B, K, Q, C], col_ok [B, K] bool -> [B, K, Q] bool
    (MASK_RULE)."""
    c = CH[name]
    if c >= zq.shape[-1]:
        raise ValueError(f"stats_mask: channel {name!r} not in a zq of {zq.shape[-1]} channels")
    v = zq[..., c]
    m = col_ok.bool()[..., None] & torch.isfinite(v) & ~(zq[..., UNK] > 0.5)
    if name in B_CHANNELS:
        m = m & (v > ZQ_FLOOR + 1e-6)
    return m


class ZStatsAcc:
    """float64 running n / sum / sum of squares / min / max per channel + the floor share of the B channels (known,
    valid states at the floor)."""

    def __init__(self, names: Sequence[str]):
        self.names = tuple(names)
        z = lambda: {n: 0.0 for n in self.names}                                        # noqa: E731
        self.n, self.s, self.ss, self.n_floor, self.n_known = z(), z(), z(), z(), z()
        self.lo = {n: math.inf for n in self.names}
        self.hi = {n: -math.inf for n in self.names}

    @torch.no_grad()
    def update(self, zq: Tensor, col_ok: Tensor) -> None:
        zq = zq.detach().double().cpu()
        col_ok = col_ok.detach().bool().cpu()
        for n in self.names:
            m = stats_mask(zq, col_ok, n)
            x = zq[..., CH[n]][m]
            self.n[n] += float(x.numel())
            self.s[n] += float(x.sum())
            self.ss[n] += float((x * x).sum())
            if x.numel():
                self.lo[n] = min(self.lo[n], float(x.min()))
                self.hi[n] = max(self.hi[n], float(x.max()))
            if n in B_CHANNELS:
                known = col_ok[..., None] & torch.isfinite(zq[..., CH[n]]) & ~(zq[..., UNK] > 0.5)
                self.n_known[n] += float(known.sum())
                self.n_floor[n] += float((known & ~m).sum())

    def channels(self, n_min: int = N_MIN, allow_small: bool = False) -> Dict[str, Dict[str, Any]]:
        out = {}
        for n in self.names:
            k = self.n[n]
            if k < n_min and not allow_small:
                raise ValueError(f"channel {n}: {int(k)} states < N_MIN {n_min}")
            mu = self.s[n] / k if k else float("nan")
            var = max(self.ss[n] / k - mu * mu, 0.0) * (k / (k - 1)) if k > 1 else float("nan")
            sd = math.sqrt(var) if var == var else float("nan")
            rec = {"mean": mu, "std": max(sd, STD_FLOOR) if sd == sd else float("nan"), "std_raw": sd, "n": int(k),
                   "min": self.lo[n] if k else None, "max": self.hi[n] if k else None,
                   "status": "ok" if k >= n_min else "small"}
            if n in B_CHANNELS:
                rec["floor_share"] = self.n_floor[n] / self.n_known[n] if self.n_known[n] else None
                rec["floor_normalised"] = (ZQ_FLOOR - mu) / rec["std"] if k else None
            out[n] = rec
        return out


def file_sha16(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_z_stats(path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """-> (record, info {path, sha16}); ValueError on a missing / foreign file."""
    p = Path(str(path))
    if not p.is_file():
        raise FileNotFoundError(f"z-stats file {p} missing (tools/ck/v5/fit_z_stats.py)")
    rec = json.loads(p.read_text())
    if rec.get("version") != STATS_VERSION:
        raise ValueError(f"{p}: version {rec.get('version')!r} != {STATS_VERSION!r} (not a fit_z_stats record)")
    return rec, {"path": str(p), "sha16": file_sha16(p)}


def _same_path(a, b) -> bool:
    return os.path.realpath(str(a)) == os.path.realpath(str(b))


def check_z_stats(rec: Dict[str, Any], arm: str, run=None, which: Optional[str] = None,
                  ckpt_sha16: Optional[str] = None, z_stride: Optional[int] = None, allow_dry_run: bool = True,
                  where: str = "z-stats", exec_model: Optional[str] = None) -> None:
    """refuse (ValueError) a record that does not belong to the teacher it is used for: arm (T = DET, M = MAP), the
    run dir (resolved), which, the checkpoint's sha16, the grid's z_stride; an 'unfinished' record (a teacher that had
    not finished, --allow-unfinished); a channel of TEACHER_CHANNELS[arm] missing / non-finite; a dry run (fewer than
    DRY_RUN_BELOW tokens or --limit) unless allow_dry_run (the launcher's strict mode refuses those); exec_model (the
    STUDENT's V4Cfg.exec_model): the states the record read Z at (its zq_exec; older records: the teacher grid's
    exec_model, 'lqr') must be the student's -- the no-LQR 'ref' student needs records of fit_z_stats --zq-exec ref."""
    if rec.get("version") != STATS_VERSION:
        raise ValueError(f"{where}: version {rec.get('version')!r} != {STATS_VERSION!r}")
    if rec.get("arm") != arm:
        raise ValueError(f"{where}: record of arm {rec.get('arm')!r} ({rec.get('run')}), expected {arm!r} "
                         f"({TEACHER_OF_ARM.get(arm)} teacher)")
    if rec.get("unfinished"):
        raise ValueError(f"{where}: record of an UNFINISHED teacher ({rec.get('run')}, --allow-unfinished): refit "
                         f"after done.json")
    if run is not None and not _same_path(rec.get("run", ""), run):
        raise ValueError(f"{where}: record of run {rec.get('run')!r}, the teacher is {str(run)!r}")
    if which is not None and str(rec.get("which")) != str(which):
        raise ValueError(f"{where}: record of which {rec.get('which')!r}, the teacher uses {which!r}")
    if ckpt_sha16 is not None and rec.get("ckpt_sha16") != ckpt_sha16:
        raise ValueError(f"{where}: record of checkpoint sha16 {rec.get('ckpt_sha16')!r}, the teacher's is "
                         f"{ckpt_sha16!r} (refit tools/ck/v5/fit_z_stats.py)")
    if z_stride is not None and int((rec.get("grid") or {}).get("z_stride", -1)) != int(z_stride):
        raise ValueError(f"{where}: record z_stride {(rec.get('grid') or {}).get('z_stride')!r} != {z_stride}")
    if exec_model is not None:
        got = rec.get("zq_exec") or (rec.get("grid") or {}).get("exec_model") or "lqr"
        if got != exec_model:
            raise ValueError(f"{where}: record read Z at {got!r} states, the student's exec_model is {exec_model!r} "
                             f"(refit tools/ck/v5/fit_z_stats.py --zq-exec {exec_model})")
    chans = rec.get("channels") or {}
    for n in TEACHER_CHANNELS[arm]:
        c = chans.get(n)
        if c is None or not (isinstance(c.get("mean"), (int, float)) and math.isfinite(c["mean"])
                             and isinstance(c.get("std"), (int, float)) and math.isfinite(c["std"])
                             and c["std"] > 0):
            raise ValueError(f"{where}: channel {n!r} missing or not finite ({c})")
    if not allow_dry_run and rec.get("dry_run", True):
        raise ValueError(f"{where}: a DRY-RUN record (n_tokens {(rec.get('data') or {}).get('n_tokens')}, limit "
                         f"{(rec.get('data') or {}).get('limit')}): run the full fit (tools/ck/v5/fit_z_stats.py "
                         f"--tokens 4096)")


def teacher_ckpt(run, which: str) -> Path:
    """the checkpoint file of a CK run (= model._ckpt_path: ckpt_<which>.pt or a .pt path)."""
    p = Path(str(which))
    if p.suffix == ".pt" and p.is_file():
        return p
    return Path(str(run)) / f"ckpt_{which}.pt"


# ----------------------------------------------------------------------------------------------- adapters
class _Part(nn.Module):
    """per-channel affine + one MLP (n + 1 -> hidden -> n, zero-initialised last layer) on [Z~_j, unknown_j]."""

    def __init__(self, n: int, hidden: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n))
        self.beta = nn.Parameter(torch.zeros(n))
        self.mlp = nn.Sequential(nn.Linear(n + 1, hidden), nn.GELU(), nn.Linear(hidden, n))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z: Tensor, unk: Tensor) -> Tensor:
        return self.gamma * z + self.beta + self.mlp(torch.cat([z, unk], -1))


class ZAdapter(nn.Module):
    """P_det / P_map of the student (module docstring).  n_z = the corrector's zq width (4, or 6 with the relation
    H); z_channels = V5Cfg.z_channels ('all' builds both halves, 'det' P_det only, 'map' P_map only)."""

    def __init__(self, n_z: int, hidden: int = 32, z_channels: str = "all"):
        super().__init__()
        if z_channels not in ("all", "det", "map"):
            raise ValueError(f"ZAdapter: z_channels {z_channels!r}")
        self.n_z = int(n_z)
        self.det_ch = tuple(c for c in DET_CH if c < self.n_z)
        self.map_ch = MAP_CH
        self.register_buffer("z_mu", torch.zeros(self.n_z))
        self.register_buffer("z_sd", torch.ones(self.n_z))
        self.register_buffer("z_fitted", torch.zeros(2, dtype=torch.uint8))        # [det, map] statistics set
        self.p_det = _Part(len(self.det_ch), int(hidden)) if z_channels in ("all", "det") else None
        self.p_map = _Part(len(self.map_ch), int(hidden)) if z_channels in ("all", "map") else None

    def _need(self) -> Tuple[bool, bool]:
        return self.p_det is not None, self.p_map is not None

    def fitted(self) -> bool:
        nd, nm = self._need()
        return (not nd or bool(self.z_fitted[0])) and (not nm or bool(self.z_fitted[1]))

    def _check(self) -> None:
        if not self.fitted():
            raise ValueError("ZAdapter: the teacher statistics are not set (z_fitted "
                             f"{self.z_fitted.tolist()}): ZAdapter.set_stats(<fit_z_stats DET record>, <MAP record>) "
                             "or load a checkpoint that carries them (no silent identity)")

    @torch.no_grad()
    def set_stats(self, det_rec: Optional[Dict[str, Any]], map_rec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """fill z_mu / z_sd from the fit_z_stats records (DET: b_nc, b_ttc, H_nc, H_ttc within n_z; MAP: d_min); a
        built half needs its record.  -> {channel: (mu, sd)}."""
        nd, nm = self._need()
        out = {}
        for want, rec, arm, flag in ((nd, det_rec, "T", 0), (nm, map_rec, "M", 1)):
            if not want:
                continue
            if rec is None:
                raise ValueError(f"ZAdapter.set_stats: the {TEACHER_OF_ARM[arm]} record is required (P_"
                                 f"{TEACHER_OF_ARM[arm]} is built)")
            check_z_stats(rec, arm, where=f"ZAdapter.set_stats {TEACHER_OF_ARM[arm]}")
            for n in TEACHER_CHANNELS[arm]:
                c = CH[n]
                if c >= self.n_z:
                    continue
                mu = float(rec["channels"][n]["mean"])
                sd = max(float(rec["channels"][n]["std"]), STD_FLOOR, SD_FLOOR_CH.get(n, 0.0))
                self.z_mu[c] = mu
                self.z_sd[c] = sd
                out[n] = (mu, sd)
            self.z_fitted[flag] = 1
        return out

    def normalise(self, z: Tensor) -> Tensor:
        """(z - mu_T) / sigma_T per channel of z [..., C] (C <= n_z: a MAP teacher's 4-channel zq too); the unknown flag
        (mu 0, sd 1) and the channels of an unbuilt half are unchanged bit for bit.  Float32, autocast off."""
        self._check()
        C = int(z.shape[-1])
        if C > self.n_z:
            raise ValueError(f"ZAdapter.normalise: {C} channels > n_z {self.n_z}")
        with torch.autocast(device_type=z.device.type, enabled=False):
            return (z.float() - self.z_mu[:C].to(z.device)) / self.z_sd[:C].to(z.device)

    def forward(self, zq: Tensor) -> Tensor:
        """zq [..., n_z] (raw student reads) -> the adapted reads in the teachers' normalised units, same layout."""
        if int(zq.shape[-1]) != self.n_z:
            raise ValueError(f"ZAdapter: zq has {zq.shape[-1]} channels, expected {self.n_z}")
        with torch.autocast(device_type=zq.device.type, enabled=False):
            z = self.normalise(zq)
            unk = z[..., UNK:UNK + 1]
            cols = list(z.unbind(-1))
            for part, ch in ((self.p_det, self.det_ch), (self.p_map, self.map_ch)):
                if part is None:
                    continue
                y = part(z[..., list(ch)], unk)
                for j, c in enumerate(ch):
                    cols[c] = y[..., j]
            return torch.stack(cols, -1)
