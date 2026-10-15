"""CK Phase 2: v2 + CK end-to-end (report 45; contract ck/phase2/spec/contract_e2e.json 'model').

Used ONLY when ParaSSRConfig.ck_e2e is enabled; with the default (empty dict / enabled false) the agent never builds
anything from here (v2 bit-identical).

Schedule (epoch = trainer.current_epoch, epoch_frac = epoch + batch_idx / num_training_batches)
  phase 'replay'        epoch < record_from_epoch                      : Phase 1 r34 candidates / official labels /
                                                                         offline KD targets (targets ck_p1_*)
  phase 'replay_record' record_from_epoch <= epoch < onpolicy_from     : replay loss + the online teachers run on the
                                                                         student's current top-16 only to RECORD
                                                                         (top-16, tau'_KD) for the background labeler
  phase 'onpolicy'      epoch >= onpolicy_from_epoch                   : tau = student top-16 (detached), online teacher
                                                                         KD, GT = LabelStore (generation <= e - lag)
  recording: record_from_epoch <= epoch <= record_until_epoch (None -> max_epochs - 2)
  r = 1 in replay; on-policy r = clip((epoch_frac - onpolicy_from) / kd_ramp_epochs, 0, 1)
Student CK loss (outside the v2 grad balancer, added after it in ParaSSRAgent.compute_loss)
  L_CK = loss_weight * [ lambda_score * BCE(s(G[:, :16]), y) + lambda_corr_score * BCE(s(G[:, 16:]), y)
                         + lambda_sur * L_sur(decode(tau, z, w; slope 0.1))
                         + r * lambda_kd_score * (KL(p_T || s(tau)) + KL(p_T,corr || s(tau'_KD)))
                         + r * w_ema * ctrl_kd_l1(c_lon, e_lat; DET c_lon, MAP e_lat; 0.25 / 1.0) ]
         + 0.0 * (gate_logit.sum() + z_lon.sum() + w_lat.sum())                (every student parameter in the graph)
  replay: tau = ck_p1_cand, tau'_KD = ck_p1_kdc, G = (ck_p1_cand, ck_p1_kdc) -- the same objective as
  tools/ck/train_ck.compute_loss (kd on, corr_aug on, ema) on the same inputs.
  w_ema = ck.losses.EmaBalancer (rank-local, no collective), updated every micro-batch with (lambda_sur * L_sur, L_kd).
Gradient paths: every trajectory input is detached (tau, tau'_KD, G); bev_embed -> AdapterS with gradient scale
  bev_grad_scale; teachers no_grad, frozen, not registered (plain python object), fp16 autocast (teacher_amp).
Non-finite: a non-finite L_CK is replaced by 0 * sum(student params) (counted, ck/loss_nonfinite); a non-finite
  gradient skips the whole optimiser step (callback, StageE rule).  CK params are clipped to `clip` first, then
  Lightning's global clip.
"""
from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .constants import CK_KEYS, CK_LABEL_IDX, KD_CTRL_W, SUR_MARGINS, SUR_TERMS, SUR_WEIGHTS

PHASES = ("replay", "replay_record", "onpolicy")
PHASE_CODE = {p: i for i, p in enumerate(PHASES)}
N_ROWS = 85109                       # packed/navtrain_train rows (e2e_data.N_ROWS)
SIM_IDX = {"nc": 0, "dac": 1, "ttc": 3}   # targets['sim_reward'] key order = CK_KEYS
RAW_FAIL_R34 = {"nc": 0.024, "dac": 0.034, "ttc": 0.074}   # r34 navtrain_train reference (report 45 §4-3)
_CK = "/home/external-user/ssd/yongjae_refiner/ck"


# ----------------------------------------------------------------------------------------------- config
def _default_phase1() -> Dict[str, str]:
    return {"packed": f"{_CK}/packed/navtrain_train", "labels_cand": f"{_CK}/labels/navtrain_train/cand",
            "labels_kd_corr": f"{_CK}/labels/navtrain_train/kd_corr", "kd_targets": f"{_CK}/kd_targets/phase1"}


_NESTED = {
    "kd_ctrl_w": lambda: dict(KD_CTRL_W),
    "kd_ema": lambda: {"ratio": 1.0, "m": 0.99, "floor": 1e-4, "cap": 10.0, "start_mb": 500},
    "phase1": _default_phase1,
}


@dataclass
class CKE2EConfig:
    enabled: bool = False
    io_dir: str = ""
    topk: int = 16
    teacher_det_run: str = f"{_CK}/train/ckT_p1"
    teacher_map_run: str = f"{_CK}/train/ckM_p1"
    teacher_which: str = "last"
    teacher_amp: bool = True
    teacher_rescore_kd: bool = True
    amp: bool = False
    bev_grad_scale: float = 1.0
    seed: int = 0
    score_prior: str = "phase1"
    loss_weight: float = 1.0
    lambda_score: float = 1.0
    lambda_corr_score: float = 1.0
    lambda_kd_score: float = 0.5
    lambda_sur: float = 1.0
    kd_ctrl_w: Dict[str, float] = field(default_factory=_NESTED["kd_ctrl_w"])
    kd_ema: Dict[str, Any] = field(default_factory=_NESTED["kd_ema"])
    lon_st_slope: float = 0.1
    lr_mult: float = 3.0
    weight_decay: float = 0.01
    clip: float = 1.0
    replay_until_epoch: int = 5          # informational (phase() uses record_from / onpolicy_from; see deviations)
    record_from_epoch: int = 4
    record_until_epoch: Optional[int] = None
    onpolicy_from_epoch: int = 5
    kd_ramp_epochs: float = 3.0
    label_lag: int = 1
    label_refresh_every_mb: int = 500
    phase1: Dict[str, str] = field(default_factory=_default_phase1)
    rec_chunk_tokens: int = 64
    grad_share_every: int = 50
    log_every: int = 1
    infer_outputs: bool = True
    strict_rows: float = 0.999

    # ------------------------------------------------------------------------------------------ parse
    @classmethod
    def from_any(cls, x: Any) -> "CKE2EConfig":
        """None / dict / DictConfig / CKE2EConfig -> CKE2EConfig (defaults merged, nested dicts merged key-wise);
        an unknown key (top level or nested) is a ValueError."""
        if isinstance(x, CKE2EConfig):
            return cls.from_any(x.to_dict())
        if x is None:
            x = {}
        try:  # DictConfig (Hydra without _convert_)
            from omegaconf import DictConfig, OmegaConf
            if isinstance(x, DictConfig):
                x = OmegaConf.to_container(x, resolve=True)
        except ImportError:  # pragma: no cover
            pass
        if not isinstance(x, dict):
            raise ValueError(f"ck_e2e must be a mapping, got {type(x).__name__}")
        known = {f.name: f for f in fields(cls)}
        unknown = sorted(set(x) - set(known))
        if unknown:
            raise ValueError(f"ck_e2e: unknown key(s) {unknown}")
        kw: Dict[str, Any] = {}
        for k, v in x.items():
            if k in _NESTED:
                base = _NESTED[k]()
                if v is None:
                    v = {}
                if not isinstance(v, dict):
                    raise ValueError(f"ck_e2e.{k} must be a mapping, got {v!r}")
                bad = sorted(set(v) - set(base))
                if bad:
                    raise ValueError(f"ck_e2e.{k}: unknown key(s) {bad}")
                base.update(v)
                kw[k] = base
            else:
                kw[k] = v
        c = cls(**kw)
        c._coerce()
        return c

    def _coerce(self) -> None:
        b = lambda v: v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")  # noqa: E731
        for k in ("enabled", "teacher_amp", "teacher_rescore_kd", "amp", "infer_outputs"):
            setattr(self, k, b(getattr(self, k)))
        for k in ("topk", "seed", "replay_until_epoch", "record_from_epoch", "onpolicy_from_epoch", "label_lag",
                  "label_refresh_every_mb", "rec_chunk_tokens", "grad_share_every", "log_every"):
            setattr(self, k, int(getattr(self, k)))
        if self.record_until_epoch is not None:
            self.record_until_epoch = int(self.record_until_epoch)
        for k in ("bev_grad_scale", "loss_weight", "lambda_score", "lambda_corr_score", "lambda_kd_score",
                  "lambda_sur", "lon_st_slope", "lr_mult", "weight_decay", "clip", "kd_ramp_epochs", "strict_rows"):
            setattr(self, k, float(getattr(self, k)))
        self.kd_ctrl_w = {k: float(v) for k, v in self.kd_ctrl_w.items()}
        e = dict(self.kd_ema)
        self.kd_ema = {"ratio": float(e["ratio"]), "m": float(e["m"]), "floor": float(e["floor"]),
                       "cap": None if e["cap"] is None else float(e["cap"]), "start_mb": int(e["start_mb"])}
        self.io_dir = "" if self.io_dir is None else str(self.io_dir)
        self.teacher_map_run = "" if self.teacher_map_run in (None, "none", "None") else str(self.teacher_map_run)
        self.teacher_det_run = str(self.teacher_det_run)
        self.phase1 = {k: str(v) for k, v in self.phase1.items()}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # ------------------------------------------------------------------------------------------ validate
    def validate(self, parent=None) -> None:
        """Contract config.validation (only when enabled).  parent: the ParaSSRConfig."""
        if not self.enabled:
            return
        if parent is not None:
            if getattr(parent, "refiner_mode", "off") != "off":
                raise ValueError("ck_e2e needs refiner_mode off (Stage E and CK e2e are exclusive)")
            if not getattr(parent, "plan_anchor", False):
                raise ValueError("ck_e2e needs plan_anchor=true (top-16 of the anchor planner)")
            if (int(parent.bev_h), int(parent.bev_w)) != (50, 100) or tuple(float(v) for v in parent.pc_range) != (
                    -32.0, 0.0, -2.0, 32.0, 32.0, 2.0):
                raise ValueError("ck_e2e reads the 50 x 100 S grid (0.64 m, front ROI)")
            if int(parent.embed_dims) != 256:
                raise ValueError(f"ck_e2e needs embed_dims 256, got {parent.embed_dims}")
        if not 1 <= self.topk <= 256:
            raise ValueError(f"ck_e2e.topk must be in [1, 256], got {self.topk}")
        if self.record_from_epoch > self.onpolicy_from_epoch:
            raise ValueError(f"ck_e2e: record_from_epoch {self.record_from_epoch} > onpolicy_from_epoch "
                             f"{self.onpolicy_from_epoch} (on-policy needs recorded labels)")
        if self.label_lag < 1:
            raise ValueError(f"ck_e2e.label_lag must be >= 1, got {self.label_lag}")
        if not self.io_dir or not os.path.isabs(self.io_dir) or "=" in self.io_dir:
            raise ValueError(f"ck_e2e.io_dir must be an absolute path without '=', got {self.io_dir!r}")
        if self.score_prior not in ("phase1", "none"):
            raise ValueError(f"ck_e2e.score_prior must be phase1|none, got {self.score_prior!r}")
        if set(self.kd_ctrl_w) != {"lon", "lat"}:
            raise ValueError(f"ck_e2e.kd_ctrl_w needs lon / lat, got {self.kd_ctrl_w}")
        if not (0.0 < self.kd_ema["m"] < 1.0 and self.kd_ema["floor"] > 0.0 and self.kd_ema["ratio"] >= 0.0):
            raise ValueError(f"ck_e2e.kd_ema needs 0 < m < 1, floor > 0, ratio >= 0, got {self.kd_ema}")
        if self.rec_chunk_tokens < 1 or self.log_every < 1:
            raise ValueError("ck_e2e.rec_chunk_tokens and log_every must be >= 1")
        if not 0.0 <= self.strict_rows <= 1.0:
            raise ValueError(f"ck_e2e.strict_rows must be in [0, 1], got {self.strict_rows}")
        # teacher arms (checked when the run dir is there; a missing run fails at the first training step)
        for key, want in (("teacher_det_run", "T"), ("teacher_map_run", "M")):
            run = getattr(self, key)
            if not run:
                if key == "teacher_det_run":
                    raise ValueError("ck_e2e.teacher_det_run is required")
                continue
            cj = Path(run) / "config.json"
            if cj.is_file():
                arm = json.loads(cj.read_text()).get("arm")
                if arm != want:
                    raise ValueError(f"ck_e2e.{key} {run}: config.json arm {arm!r}, expected {want!r}")


# ----------------------------------------------------------------------------------------------- student / prior
def phase1_label_prior(labels_cand_dir: str, n_max: int = 20000) -> Optional[np.ndarray]:
    """Mean official label per CK key over ok candidates of <= n_max evenly spaced rows of labels/<split>/cand
    (= tools/ck/train_ck.label_prior); None if the files are not there."""
    d = Path(labels_cand_dir)
    if not (d / "labels.npy").is_file() or not (d / "ok.npy").is_file():
        return None
    lab = np.load(d / "labels.npy", mmap_mode="r")
    ok = np.load(d / "ok.npy", mmap_mode="r")
    rows = np.unique(np.linspace(0, len(lab) - 1, min(n_max, len(lab))).astype(np.int64))
    L = np.asarray(lab[rows], np.float64)[..., list(CK_LABEL_IDX)]
    m = np.asarray(ok[rows], bool) & np.isfinite(L).all(-1)
    if not m.any():
        return None
    return np.clip(L[m].mean(0), 1e-3, 1 - 1e-3)


def build_student_ck(ckcfg: CKE2EConfig, apply_prior: bool = True):
    """CKNet('S', seed, score_hidden 256, lead_aux False) with adapter.bev_grad_scale = g; score-head bias =
    logit(Phase 1 label mean) when score_prior == 'phase1' and apply_prior (skipped when a checkpoint will be
    loaded: no file is opened then).  No global RNG is consumed."""
    from .model import CKNet

    net = CKNet("S", int(ckcfg.seed), score_hidden=256, lead_aux=False)
    net.trunk.adapter.bev_grad_scale = float(ckcfg.bev_grad_scale)
    net.init_report = {"source": None, "score_prior": None}
    if apply_prior and ckcfg.score_prior == "phase1":
        prior = phase1_label_prior(ckcfg.phase1["labels_cand"])
        if prior is not None:
            net.set_score_prior(prior)
            net.init_report["score_prior"] = [float(v) for v in prior]
    return net


# ----------------------------------------------------------------------------------------------- tensors
def student_topk(predictions: Dict[str, torch.Tensor], k: int) -> Dict[str, torch.Tensor]:
    """CK candidates = (anchors + offset).gather(plan_final_rewards.topk(k)) (dump_v2 gather), all detached ->
    cand f32 [B,K,8,3], idx int64 [B,K], final [B,K], im [B,K], sim [B,K,5]."""
    final = predictions["plan_final_rewards"].detach().float()
    idx = final.topk(int(k), dim=-1).indices
    off = predictions["trajectory_offset"].detach().float()
    refined = predictions["trajectory_anchors"].detach().float().unsqueeze(0) + off
    T, P = refined.shape[-2:]
    cand = refined.gather(1, idx[:, :, None, None].expand(-1, -1, T, P))
    im = predictions["im_rewards"].detach().float().gather(1, idx)
    sim = predictions["sim_rewards"].detach().float()                                  # [B, 5, 256]
    sim = sim.gather(2, idx[:, None, :].expand(-1, sim.shape[1], -1)).transpose(1, 2)  # [B, K, 5]
    return {"cand": cand.contiguous(), "idx": idx, "final": final.gather(1, idx), "im": im, "sim": sim.contiguous()}


def bev_sgrid(bev_embed: torch.Tensor) -> torch.Tensor:
    """bev_embed [B, 5000, 256] -> S grid [B, 256, 50, 100] (gradient kept; AdapterSGrid flattens it back)."""
    B, HW, C = bev_embed.shape
    if HW != 50 * 100:
        raise ValueError(f"bev_embed {tuple(bev_embed.shape)} is not the 50 x 100 grid")
    return bev_embed.transpose(1, 2).reshape(B, C, 50, 100)


def _sanitize_traj(traj: torch.Tensor, fallback: torch.Tensor, ok: Optional[torch.Tensor] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """traj [B, K, 8, 3]: candidates that are non-finite (or not ok) -> fallback [B, 8, 3] (finite) so the student
    never sees NaN geometry (a NaN logit times a 0 mask is still NaN).  -> (traj, finite mask [B, K])."""
    fin = torch.isfinite(traj).flatten(2).all(-1)
    good = fin if ok is None else fin & ok.bool()
    t = torch.where(good[..., None, None], torch.nan_to_num(traj, nan=0.0, posinf=0.0, neginf=0.0),
                    fallback[:, None].to(traj.dtype).expand_as(traj))
    return t, fin


# ----------------------------------------------------------------------------------------------- teachers
class TeacherPair:
    """Frozen DET (arm T) + MAP (arm M) CK teachers (ck.model.load_ck), run online on the student's candidates.
    Plain python object (not an nn.Module attribute of the agent: not in the state_dict / DDP / optimiser)."""

    def __init__(self, det_run: str, map_run: str = "", which: str = "last", device="cpu", amp: bool = True):
        from .model import load_ck

        self.device = torch.device(device)
        self.det, self.det_cfg = load_ck(det_run, which, device=str(self.device))
        self.map, self.map_cfg = (load_ck(map_run, which, device=str(self.device)) if map_run else (None, None))
        if self.det_cfg.get("arm") != "T" or (self.map is not None and self.map_cfg.get("arm") != "M"):
            raise ValueError(f"teacher arms must be T / M, got {self.det_cfg.get('arm')} / "
                             f"{None if self.map_cfg is None else self.map_cfg.get('arm')}")
        for n in self.nets():
            n.requires_grad_(False)
            n.eval()
        self.amp = bool(amp)

    def nets(self):
        return [n for n in (self.det, self.map) if n is not None]

    def to(self, device) -> "TeacherPair":
        device = torch.device(device)
        if device != self.device:
            for n in self.nets():
                n.to(device)
            self.device = device
        return self

    @staticmethod
    def _ok(ok_tok, score, c, e, K):
        ok = ok_tok.reshape(-1).bool().to(score.device)[:, None].expand(-1, K)
        return ok & torch.isfinite(score).all(-1) & torch.isfinite(c).all(-1) & torch.isfinite(e).all(-1)

    @torch.no_grad()
    def run(self, bev_det, ok_det, bev_map, ok_map, cand, status, rescore: bool) -> Dict[str, torch.Tensor]:
        """bev_* [B, 256, 50, 100] (f16 cache, -> float), ok_* [B], cand [B, K, 8, 3], status [B, 8] ->
        kd_score_prob [B,K,5], kd_ok [B,K], kd_c_lon / kd_e_lat [B,K,6], kd_corr [B,K,8,3] and (rescore)
        kd_score_prob_corr [B,K,5], kd_ok_corr [B,K] (zeros / False without rescore).  Same rule as Phase 1
        tools/ck/kd_targets.py: each teacher's own decode (slope 0) -> combine_teacher; tau'_KD = correct(cand,
        z_DET, w_MAP, v0, 0) (w_DET where the MAP BEV is missing); tau'_KD rescored by both teachers (scene features
        reused) -> combine_teacher."""
        from ..refiner.e2e import ego_inputs
        from .correct import correct
        from .kd import combine_teacher

        dev = self.device
        cand = cand.detach().float().to(dev)
        T, K = cand.shape[:2]
        ego = ego_inputs(status.detach().float().to(dev))
        v0 = ego[0]
        ac = torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=self.amp and dev.type == "cuda")
        outs, scenes = [], []
        for net, bev in ((self.det, bev_det), (self.map, bev_map)):
            if net is None:
                continue
            with ac:
                feat, mem = net.trunk.scene(bev.to(dev).float(), T)
                o = net.trunk.candidates(feat, mem, cand, *ego)
                s = net.score(o, T, K)
            outs.append({"score_logit": s.float(), "z": o["z_lon"].float(), "w": o["w_lat"].float()})
            scenes.append((net, feat, mem))
        od = outs[0]
        # ONE batched decode (correct() is launch-bound, ~18 ms whatever K; candidates are decoded independently):
        #   mode A: c_lon depends on z only -> DET c_lon = c_lon of decode(z_DET, w_src); e_lat depends on (z, w) ->
        #   the MAP teacher's own e_lat needs decode(z_MAP, w_MAP); tau'_KD = decode(z_DET, w_src)['traj'] with
        #   w_src = w_MAP (w_DET where the MAP BEV is missing: then its e_lat is the DET teacher's own, the
        #   combine_teacher fallback).  Same values as Phase 1 kd_targets (3 separate decodes).
        w_src = od["w"]
        if self.map is not None:
            om = outs[1]
            tok_map = ok_map.reshape(-1).bool().to(dev)
            w_src = torch.where(tok_map[:, None, None], om["w"], od["w"])
            dec = correct(torch.cat([cand, cand], 1), torch.cat([om["z"], od["z"]], 1),
                          torch.cat([om["w"], w_src], 1), v0, 0.0)
            sl = slice(K, 2 * K)
            om["c_lon"], om["e_lat"] = dec["c_lon"][:, :K, 2:], dec["e_lat"][:, :K, 2:]
        else:
            dec = correct(cand, od["z"], w_src, v0, 0.0)
            sl = slice(0, K)
        od["c_lon"], od["e_lat"] = dec["c_lon"][:, sl, 2:], dec["e_lat"][:, sl, 2:]
        kd_corr = dec["traj"][:, sl].float()
        det = {"score_logit": od["score_logit"], "c_lon": od["c_lon"], "e_lat": od["e_lat"],
               "ok": self._ok(ok_det, od["score_logit"], od["c_lon"], od["e_lat"], K)}
        mp = None
        if self.map is not None:
            mp = {"score_logit": om["score_logit"], "c_lon": om["c_lon"], "e_lat": om["e_lat"],
                  "ok": self._ok(ok_map, om["score_logit"], om["c_lon"], om["e_lat"], K)}
        comb = combine_teacher(det, mp)
        kd_ok = comb["kd_ok"].bool() & torch.isfinite(kd_corr).flatten(2).all(-1)
        res = {"kd_score_prob": comb["kd_score_prob"].float(), "kd_ok": kd_ok, "kd_c_lon": comb["kd_c_lon"].float(),
               "kd_e_lat": comb["kd_e_lat"].float(), "kd_corr": kd_corr}
        if rescore:
            kc = torch.nan_to_num(kd_corr, nan=0.0)
            sc = []
            for net, feat, mem in scenes:
                with ac:
                    sc.append(net.score(net.trunk.candidates(feat, mem, kc, *ego), T, K).float())
            okc = lambda tok, s: tok.reshape(-1).bool().to(dev)[:, None].expand(-1, K) & torch.isfinite(s).all(-1)  # noqa: E731
            detc = {"score_logit": sc[0], "ok": okc(ok_det, sc[0])}
            mpc = {"score_logit": sc[1], "ok": okc(ok_map, sc[1])} if self.map is not None else None
            combc = combine_teacher(detc, mpc)
            res["kd_score_prob_corr"] = combc["kd_score_prob"].float()
            res["kd_ok_corr"] = kd_ok & combc["kd_ok"].bool()
        else:
            res["kd_score_prob_corr"] = torch.zeros_like(res["kd_score_prob"])
            res["kd_ok_corr"] = torch.zeros_like(kd_ok)
        return res


def student_forward(student, bev: torch.Tensor, cand: torch.Tensor, status: torch.Tensor, extra: torch.Tensor,
                    slope: float) -> Dict:
    """CKNet.forward(bev, cand, status, extra=extra, decode=True, slope) with the candidate and extra sets run through
    ONE trunk.candidates call (candidates are processed independently: per-candidate self-attention, per-query cross
    attention to the token memory, per-sample GroupNorm), saving one launch-bound pass in forward and backward.
    Same outputs / keys as CKNet.forward (gate_logit over all K + K2 candidates)."""
    from ..refiner.e2e import ego_inputs
    from .correct import correct

    T, K = cand.shape[:2]
    K2 = extra.shape[1]
    ego = ego_inputs(status)
    v0, a0, eds, cmd = ego
    cand = cand.float()
    feat, mem = student.trunk.scene(bev, T)
    o = student.trunk.candidates(feat, mem, torch.cat([cand, extra.float()], 1), v0, a0, eds, cmd)
    sc = student.score(o, T, K + K2)
    out = {"score_logit": sc[:, :K], "extra_score_logit": sc[:, K:], "z_lon": o["z_lon"][:, :K].float(),
           "w_lat": o["w_lat"][:, :K].float(), "gate_logit": o["gate_logit"].float(), "ego": ego,
           "z_lon_all": o["z_lon"], "w_lat_all": o["w_lat"]}
    out["corr"] = correct(cand, out["z_lon"], out["w_lat"], v0, slope)
    return out


# ----------------------------------------------------------------------------------------------- recorder
class CandRecorder:
    """Per-rank, per-epoch writer of the student's (top-16, tau'_KD) for the background labeler: chunks of
    chunk_tokens tokens -> e2e_data.write_rec_chunk (tmp + os.replace) at rec_chunk_path(io_dir, epoch, rank,
    attempt, seq); close_epoch() flushes the rest and writes DONE_<attempt>.json (normal epoch end only).
    attempt = 8 hex chars, new for every recorder (process x epoch start)."""

    def __init__(self, io_dir: str, rank: int, world_size: int, epoch: int, chunk_tokens: int = 64,
                 attempt: Optional[str] = None):
        self.io_dir, self.rank, self.world_size, self.epoch = str(io_dir), int(rank), int(world_size), int(epoch)
        self.chunk_tokens = int(chunk_tokens)
        self.attempt = attempt or uuid.uuid4().hex[:8]
        self.seq, self.n_tokens, self.closed = 0, 0, False
        self.chunks: List[str] = []
        self._buf: Dict[str, List[np.ndarray]] = {k: [] for k in ("row", "cand", "cand_idx", "kd_corr", "kd_ok",
                                                                  "gstep")}
        self._n_buf = 0

    def add(self, rows, cand, cand_idx, kd_corr, kd_ok, gstep: int) -> int:
        """rows int64 [n] (row < 0 skipped), cand f32 [n,16,8,3], cand_idx [n,16], kd_corr f32 [n,16,8,3],
        kd_ok bool [n,16] -> number of tokens buffered."""
        if self.closed:
            raise RuntimeError("CandRecorder.add after close_epoch")
        a = lambda x: x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)  # noqa: E731
        r = a(rows).reshape(-1).astype(np.int64)
        keep = r >= 0
        n = int(keep.sum())
        if n == 0:
            return 0
        self._buf["row"].append(r[keep])
        self._buf["cand"].append(a(cand)[keep].astype(np.float32))
        self._buf["cand_idx"].append(a(cand_idx)[keep].astype(np.int16))
        self._buf["kd_corr"].append(a(kd_corr)[keep].astype(np.float32))
        self._buf["kd_ok"].append(a(kd_ok)[keep].astype(bool))
        self._buf["gstep"].append(np.full(n, int(gstep), np.int64))
        self._n_buf += n
        while self._n_buf >= self.chunk_tokens:
            self._write(self.chunk_tokens)
        return n

    def _write(self, n: int) -> None:
        from . import e2e_data as D

        cat = {k: np.concatenate(v, 0) for k, v in self._buf.items()}
        take = {k: v[:n] for k, v in cat.items()}
        rest = {k: v[n:] for k, v in cat.items()}
        path = Path(D.rec_chunk_path(self.io_dir, self.epoch, self.rank, self.attempt, self.seq))
        path.parent.mkdir(parents=True, exist_ok=True)
        D.write_rec_chunk(path, take["row"], take["cand"], take["cand_idx"], take["kd_corr"], take["kd_ok"],
                          take["gstep"], self.epoch, self.rank)
        self.chunks.append(path.name)
        self.seq += 1
        self.n_tokens += n
        self._buf = {k: ([v] if len(v) else []) for k, v in rest.items()}
        self._n_buf -= n

    def flush(self) -> None:
        """write the partial chunk (no DONE)."""
        if self._n_buf > 0:
            self._write(self._n_buf)

    def close_epoch(self) -> None:
        from . import e2e_data as D

        if self.closed:
            return
        self.flush()
        D.write_rec_done(self.io_dir, self.epoch, self.rank, self.world_size, self.attempt, list(self.chunks),
                         int(self.n_tokens))
        self.closed = True


# ----------------------------------------------------------------------------------------------- loss / infer
def _f(x) -> float:
    return float(x.detach()) if torch.is_tensor(x) else float(x)


class CKE2E:
    """Loss / inference helper held by ParaSSRAgent (plain object).  epoch / epoch_frac / rank / world_size /
    gstep are set by the callback (make_ck_callback); mb counts this rank's micro-batches."""

    def __init__(self, cfg: CKE2EConfig, max_epochs: int = 30):
        from .losses import EmaBalancer

        self.cfg = CKE2EConfig.from_any(cfg)
        self.max_epochs = int(max_epochs)
        e = self.cfg.kd_ema
        self.ema = EmaBalancer(ratio=e["ratio"], m=e["m"], floor=e["floor"], cap=e["cap"], start_step=e["start_mb"])
        self.epoch, self.epoch_frac, self.rank, self.world_size, self.batch_idx, self.gstep = 0, 0.0, 0, 1, 0, 0
        self.mb = 0
        self.skipped_steps = 0
        self.cum = {"tokens": 0, "rows_ok": 0, "nonfinite_loss": 0, "sur_nonfinite": 0, "rec_tokens": 0,
                    "G_src_phase1": 0, "G_src_prev": 0, "G_src_older": 0}
        self._teachers: Optional[TeacherPair] = None
        self._labels = None
        self._labels_max_epoch: Optional[int] = None
        self.recorder: Optional[CandRecorder] = None
        self._rows_checked = False
        self.last_clip_norm: Optional[float] = None

    # -------------------------------------------------------------------------------------- schedule
    @property
    def record_until(self) -> int:
        c = self.cfg.record_until_epoch
        return self.max_epochs - 2 if c is None else int(c)

    def phase(self, epoch: int) -> str:
        e = int(epoch)
        if e >= self.cfg.onpolicy_from_epoch:
            return "onpolicy"
        if e >= self.cfg.record_from_epoch:
            return "replay_record"
        return "replay"

    def r(self, epoch_frac: float) -> float:
        if self.phase(int(math.floor(epoch_frac))) != "onpolicy":
            return 1.0
        ramp = float(self.cfg.kd_ramp_epochs)
        if ramp <= 0:
            return 1.0
        return float(min(max((float(epoch_frac) - self.cfg.onpolicy_from_epoch) / ramp, 0.0), 1.0))

    def recording(self, epoch: int) -> bool:
        return self.cfg.record_from_epoch <= int(epoch) <= self.record_until

    # -------------------------------------------------------------------------------------- state
    def state_dict(self) -> Dict[str, Any]:
        return {"ema": self.ema.state_dict(), "mb": int(self.mb), "cum": dict(self.cum),
                "skipped_steps": int(self.skipped_steps)}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        if not sd:
            return
        self.ema.load_state_dict(sd["ema"])
        self.mb = int(sd.get("mb", 0))
        self.cum.update({k: int(v) for k, v in dict(sd.get("cum", {})).items()})
        self.skipped_steps = int(sd.get("skipped_steps", 0))

    # -------------------------------------------------------------------------------------- lazy pieces
    def teachers(self, device) -> TeacherPair:
        if self._teachers is None:
            c = self.cfg
            self._teachers = TeacherPair(c.teacher_det_run, c.teacher_map_run, c.teacher_which, device, c.teacher_amp)
        return self._teachers.to(device)

    def labels(self):
        if self._labels is None:
            from . import e2e_data as D
            self._labels = D.LabelStore(self.cfg.io_dir, dict(self.cfg.phase1), n_rows=N_ROWS)
        return self._labels

    def refresh_labels(self, epoch: Optional[int] = None) -> Dict:
        e = self.epoch if epoch is None else int(epoch)
        max_epoch = e - self.cfg.label_lag
        st = self.labels().refresh(max_epoch)
        self._labels_max_epoch = max_epoch
        return dict(st or {}, max_epoch=max_epoch)

    def _check_rows(self, rows: torch.Tensor) -> None:
        n = int(rows.numel())
        ok = int((rows >= 0).sum())
        self.cum["tokens"] += n
        self.cum["rows_ok"] += ok
        if self.cfg.strict_rows <= 0:
            return
        if self.mb == 0 and n > 0 and ok == 0:
            raise RuntimeError("ck_e2e: no training token of the first micro-batch has a packed navtrain_train row "
                               "(targets['ck_row'] all -1): wrong split / token mapping")
        if not self._rows_checked and self.cum["tokens"] >= 2000:
            self._rows_checked = True
            frac = self.cum["rows_ok"] / max(self.cum["tokens"], 1)
            if frac < self.cfg.strict_rows:
                raise RuntimeError(f"ck_e2e: only {frac:.4f} of the first {self.cum['tokens']} training tokens have a "
                                   f"packed row (< strict_rows {self.cfg.strict_rows})")

    # -------------------------------------------------------------------------------------- inference
    @torch.no_grad()
    def infer(self, student, features, predictions) -> Dict[str, torch.Tensor]:
        """eval-mode outputs (contract model.eval_outputs): student.eval(), fp32, decode slope 0, corrected
        candidates rescored with the same scene features.  predictions['trajectory'] is untouched."""
        bev = predictions["bev_embed"]
        top = student_topk(predictions, self.cfg.topk)
        status = features["status_feature"].float()
        with torch.autocast(device_type=bev.device.type, enabled=False):
            out = student(bev_sgrid(bev.float()), top["cand"], status, decode=True, slope=0.0, rescore_corr=True)
        c = out["corr"]
        return {"ck_cand": top["cand"], "ck_cand_idx": top["idx"], "ck_v2_final": top["final"], "ck_v2_im": top["im"],
                "ck_v2_sim": top["sim"], "ck_score_logit": out["score_logit"].float(), "ck_z_lon": out["z_lon"].float(),
                "ck_w_lat": out["w_lat"].float(), "ck_c_lon": c["c_lon"][..., 2:].float(),
                "ck_e_lat": c["e_lat"][..., 2:].float(), "ck_corr_traj": c["traj"].float(),
                "ck_corr_score_logit": out["corr_score_logit"].float()}

    # -------------------------------------------------------------------------------------- training loss
    def loss(self, student, features, targets, predictions, v2_loss: Optional[torch.Tensor] = None
             ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        from . import losses as L

        t0 = time.time()
        c = self.cfg
        bev = predictions["bev_embed"]
        dev = bev.device
        B = bev.shape[0]
        K = c.topk
        phase = self.phase(self.epoch)
        r = self.r(self.epoch_frac)
        recording = self.recording(self.epoch) and phase != "replay"
        status = features["status_feature"].float()
        top = student_topk(predictions, K)
        tau = top["cand"]
        fallback = tau[:, 0]
        rows = (targets["ck_row"].reshape(-1).long() if "ck_row" in targets
                else torch.full((B,), -1, dtype=torch.long, device=dev))
        self._check_rows(rows)
        human = targets["trajectory"].float()
        logs: Dict[str, float] = {}

        # ---- teachers (online): recording epochs and on-policy
        T_out = None
        t_teacher = 0.0
        if phase == "onpolicy" or recording:
            tt = time.time()
            tp = self.teachers(dev)
            has_map = tp.map is not None
            T_out = tp.run(targets["kd_bev_0"], targets["kd_ok_0"], targets["kd_bev_1"] if has_map else None,
                           targets["kd_ok_1"] if has_map else None, tau, status,
                           rescore=phase == "onpolicy" and c.teacher_rescore_kd)
            t_teacher = time.time() - tt
        n_rec = 0
        if recording and self.recorder is not None:
            n_rec = self.recorder.add(rows, tau, top["idx"], T_out["kd_corr"], T_out["kd_ok"], self.gstep)
            self.cum["rec_tokens"] += n_rec

        # ---- inputs of the student
        if phase == "onpolicy":
            G = self.labels_lookup(rows)
            G_traj, G_fin = _sanitize_traj(G["G_traj"].to(dev).float(), fallback, G["G_ok"].to(dev))
            G_y, G_ok = G["G_y"].to(dev).float(), G["G_ok"].to(dev).bool() & G_fin
            cand = tau
            kdc, kdc_fin = _sanitize_traj(T_out["kd_corr"], fallback)
            extra = torch.cat([kdc, G_traj], 1)                                   # [B, 16 + 32, 8, 3]
            y_c, ok_c, y_t, ok_t = G_y[:, :K], G_ok[:, :K], G_y[:, K:], G_ok[:, K:]
            p_T, p_Tc = T_out["kd_score_prob"], T_out["kd_score_prob_corr"]
            kd_ok, kd_ok_c = T_out["kd_ok"].bool(), T_out["kd_ok_corr"].bool() & kdc_fin
            kd_c_lon, kd_e_lat = T_out["kd_c_lon"], T_out["kd_e_lat"]
            sur_ok = torch.ones(B, dtype=torch.bool, device=dev)
            src = G["G_src"].reshape(-1).long()
        else:
            g = lambda k: targets[k].to(dev)  # noqa: E731
            p1 = g("ck_p1_ok").reshape(-1).bool()
            cand, _ = _sanitize_traj(g("ck_p1_cand").float(), fallback, p1[:, None].expand(-1, K))
            kdc, _ = _sanitize_traj(g("ck_p1_kdc").float(), fallback, p1[:, None].expand(-1, K))
            extra = kdc
            y_c, ok_c = g("ck_p1_y").float(), g("ck_p1_y_ok").bool() & p1[:, None]
            y_t, ok_t = g("ck_p1_y_kdc").float(), g("ck_p1_y_kdc_ok").bool() & p1[:, None]
            p_T, p_Tc = g("ck_p1_kd_prob").float(), g("ck_p1_kd_prob_corr").float()
            kd_ok = g("ck_p1_kd_ok").bool() & p1[:, None]
            kd_ok_c = kd_ok
            kd_c_lon, kd_e_lat = g("ck_p1_kd_c_lon").float(), g("ck_p1_kd_e_lat").float()
            sur_ok = p1
            src = torch.zeros(B, dtype=torch.long)

        # ---- student forward
        ts = time.time()
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=bool(c.amp and dev.type == "cuda")):
            out = student_forward(student, bev_sgrid(bev), cand, status, extra, c.lon_st_slope)
        s_cand = out["score_logit"].float()
        ex = out["extra_score_logit"].float()
        if phase == "onpolicy":
            s_tkd, s_Gc, s_Gt = ex[:, :K], ex[:, K:2 * K], ex[:, 2 * K:]
        else:
            s_tkd, s_Gc, s_Gt = ex, s_cand, ex

        # ---- loss (same term order as tools/ck/train_ck.compute_loss)
        l_bc, per = L.score_bce(s_Gc, y_c, ok_c)
        l_bt, _ = L.score_bce(s_Gt, y_t, ok_t)
        inner = c.lambda_score * l_bc
        inner = inner + c.lambda_corr_score * l_bt
        v0, a0 = out["ego"][0], out["ego"][1]
        ref_gt_ok = targets["ref_gt_ok"].reshape(-1).bool().to(dev) if "ref_gt_ok" in targets else \
            torch.zeros(B, dtype=torch.bool, device=dev)
        idx = L.surrogate_index(ref_gt_ok & sur_ok, human)
        n = int(idx.numel())
        l_sur = bev.new_zeros((), dtype=torch.float32)
        terms = {k: 0.0 for k in SUR_TERMS + ("P1_minus_P0",)}
        nonfin = 0
        if n > 0 and c.lambda_sur:
            ref = {k: v for k, v in targets.items() if k.startswith("ref_")}
            sb = L.surrogate_batch_k(ref, idx, cand, human, v0, a0)
            l_sur, terms, nonfin = L.correction_loss(out["corr"]["raw"], sb, n, K, SUR_WEIGHTS, SUR_MARGINS)
            inner = inner + c.lambda_sur * l_sur
        l_sur_val = _f(l_sur)
        l_ks = L.score_kd_bce(s_cand, p_T, kd_ok)
        has_corr_kd = phase != "onpolicy" or c.teacher_rescore_kd
        if has_corr_kd:
            l_ks = l_ks + L.score_kd_bce(s_tkd, p_Tc, kd_ok_c)
        inner = inner + (r * c.lambda_kd_score) * l_ks
        l_kd, parts = L.ctrl_kd_l1(out["corr"]["c_lon"][..., 2:].float(), out["corr"]["e_lat"][..., 2:].float(),
                                   kd_c_lon, kd_e_lat, kd_ok, w_lon=c.kd_ctrl_w["lon"], w_lat=c.kd_ctrl_w["lat"])
        l_kd_val = _f(l_kd)
        self.ema.update(c.lambda_sur * l_sur_val, l_kd_val)
        w = float(self.ema.weight(self.mb))
        inner = inner + (r * w) * l_kd
        loss = c.loss_weight * inner
        loss = loss + 0.0 * (out["gate_logit"].sum() + out["z_lon_all"].sum() + out["w_lat_all"].sum())
        loss_val = _f(loss)
        bad_loss = not math.isfinite(loss_val)
        if bad_loss:
            self.cum["nonfinite_loss"] += 1
            loss = 0.0 * sum(p.sum() for p in student.parameters() if p.requires_grad)
        t_student = time.time() - ts

        # ---- grad share (every grad_share_every micro-batches)
        every = int(c.grad_share_every)
        if (every > 0 and self.mb % every == 0 and v2_loss is not None and bev.requires_grad
                and torch.is_grad_enabled() and not bad_loss):
            gn = {}
            for k, t in (("v2", v2_loss), ("ck", loss)):
                gg = (torch.autograd.grad(t, bev, retain_graph=True, allow_unused=True)[0]
                      if isinstance(t, torch.Tensor) and t.requires_grad else None)
                gn[k] = 0.0 if gg is None else float(gg.float().norm())
            logs["gnorm/bev_v2"], logs["gnorm/bev_ck"] = gn["v2"], gn["ck"]
            logs["ck/gshare_ck"] = gn["ck"] / max(gn["v2"] + gn["ck"], 1e-12)

        # ---- logs
        lw = c.loss_weight
        logs.update({
            "ck/loss": loss_val if not bad_loss else 0.0, "ck/loss_nonfinite": float(bad_loss),
            "ck/bce_cand": _f(l_bc), "ck/bce_tkd": _f(l_bt), "ck/kd_score": _f(l_ks), "ck/sur": l_sur_val,
            "ck/sur_nonfinite": float(nonfin), "ck/kd_ctrl": l_kd_val, "ck/kd_lon": float(parts["lon"]),
            "ck/kd_lat": float(parts["lat"]), "ck/w_ema": w, "ck/r": r, "ck/phase": float(PHASE_CODE[phase]),
            "ck/n_gt": float(n), "ck/mb": float(self.mb),
            "ck/wt_bce_cand": lw * c.lambda_score * _f(l_bc), "ck/wt_bce_tkd": lw * c.lambda_corr_score * _f(l_bt),
            "ck/wt_kd_score": lw * r * c.lambda_kd_score * _f(l_ks), "ck/wt_sur": lw * c.lambda_sur * l_sur_val,
            "ck/wt_kd_ctrl": lw * r * w * l_kd_val,
        })
        for k, v in (per or {}).items():
            logs[f"ck/bce_{k}"] = float(v)
        for k, v in (terms or {}).items():
            logs[f"ck/t_{k}"] = float(v)
        if v2_loss is not None:
            logs["ck/ratio_v2"] = (loss_val / max(abs(_f(v2_loss)), 1e-12)) if not bad_loss else 0.0
        with torch.no_grad():
            logs["ck/kd_ok_frac"] = float(kd_ok.float().mean())
            if T_out is not None:
                logs["ck/teacher_kd_ok_frac"] = float(T_out["kd_ok"].float().mean())
            srcn = src.reshape(-1).cpu()
            for code, name in ((0, "phase1"), (1, "prev"), (2, "older")):
                cnt = int((srcn == code).sum())
                logs[f"ck/G_src_{name}"] = cnt / max(B, 1)
                self.cum[f"G_src_{name}"] += cnt
            logs["ck/G_ok_frac"] = float(torch.cat([ok_c, ok_t], 1).float().mean())
            if "sim_reward" in targets:
                sr = targets["sim_reward"].to(dev).float()                       # [B, 5, 256]
                sg = sr.gather(2, top["idx"][:, None, :].expand(-1, sr.shape[1], -1))
                vm = (targets["sim_reward_valid"].reshape(-1).to(dev).float() > 0.5 if "sim_reward_valid" in targets
                      else torch.ones(B, dtype=torch.bool, device=dev))
                den = max(int(vm.sum()) * K, 1)
                for key, j in SIM_IDX.items():
                    logs[f"ck/raw_fail_{key}"] = float(((sg[:, j] < 1.0) & vm[:, None]).sum()) / den
            cc = out["corr"]
            logs["ck/live"] = float((cc["c_lon"].detach() < 0).any(-1).float().mean())
            logs["ck/e_abs"] = float(cc["e_lat"].detach().abs().mean())
        self.cum["sur_nonfinite"] += int(nonfin)
        logs["ck/teacher_ms"] = 1e3 * t_teacher
        logs["ck/student_ms"] = 1e3 * t_student
        logs["ck/rec_tokens"] = float(n_rec)
        logs["ck/total_ms"] = 1e3 * (time.time() - t0)
        self.mb += 1
        return loss, {k: torch.tensor(float(v), device=dev) for k, v in logs.items()}

    def labels_lookup(self, rows: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self._labels_max_epoch != self.epoch - self.cfg.label_lag:
            self.refresh_labels()
        return self.labels().lookup(rows.detach().cpu().long())


# ----------------------------------------------------------------------------------------------- callback
def label_supply_warning(ck: "CKE2E", stats: Optional[Dict], min_frac: float = 0.5) -> Optional[Dict[str, Any]]:
    """Rank-0 check at an on-policy epoch start (logging only, no change to the loss): of the rows recorded in the
    label-source epoch (rec DONE n_tokens over ranks), how many already carry that generation's labels.  A dead
    labeler shows up as frac < min_frac (G silently falls back to older generations / Phase 1)."""
    if stats is None:
        return None
    src = int(stats.get("max_epoch", ck.epoch - ck.cfg.label_lag))
    if src < ck.cfg.record_from_epoch or src > ck.record_until:
        return {"warn": False, "src_epoch": src, "msg": "label-source epoch not recorded by schedule"}
    from . import e2e_data as D
    try:
        done = D.read_rec_done(ck.cfg.io_dir, src)
    except Exception as e:      # noqa: BLE001
        return {"warn": True, "src_epoch": src, "msg": f"cannot read rec DONE of epoch {src}: {e}"}
    n_rec = 0
    for ds in done.values():
        n_rec += max(int(d.get("n_tokens", 0) or 0) for d in ds)
    n_src = int(stats.get("n_prev", 0) or 0)          # rows whose newest label generation is src
    if n_rec <= 0:
        return {"warn": True, "src_epoch": src, "n_rec": 0, "n_labeled": n_src,
                "msg": f"no rec DONE for epoch {src} (recording expected): GT labels fall back to older / Phase 1"}
    frac = n_src / max(min(n_rec, N_ROWS), 1)
    out = {"warn": frac < min_frac, "src_epoch": src, "n_rec": n_rec, "n_labeled": n_src, "frac_labeled": frac}
    out["msg"] = (f"only {n_src}/{n_rec} rows recorded in epoch {src} have labels (frac {frac:.3f} < {min_frac}); "
                  f"labeler dead or slow? check 'train_e2e.sh status' / labeler/labeler.log" if out["warn"]
                  else f"{n_src}/{n_rec} recorded rows of epoch {src} labelled (frac {frac:.3f})")
    return out


def ck_parameters(agent) -> List[torch.nn.Parameter]:
    return [p for p in agent.ck_student.parameters() if p.requires_grad]


def make_ck_callback(agent):
    import pytorch_lightning as pl

    class CKE2ECallback(pl.Callback):
        """Schedule (epoch, epoch_frac, rank), recorder open / close, LabelStore refresh, the CK-only gradient clip
        and the non-finite step skip (before Lightning's global clip), rank-0 steps_rank0.jsonl / epochs.jsonl, and
        the CKE2E state (EMA, mb, counters) in the Lightning checkpoint."""

        def __init__(self):
            super().__init__()
            self._f = None
            self._n = 0
            self._t_start = self._t_end = None

        @property
        def state_key(self) -> str:
            return "CKE2ECallback"

        def state_dict(self):
            return agent._ck_e2e.state_dict()

        def load_state_dict(self, state_dict):
            agent._ck_e2e.load_state_dict(state_dict)

        # ---- helpers
        def _io(self) -> Path:
            p = Path(agent._ck_e2e.cfg.io_dir)
            p.mkdir(parents=True, exist_ok=True)
            return p

        def _append(self, name: str, rec: Dict) -> None:
            with open(self._io() / name, "a") as f:
                f.write(json.dumps(rec) + "\n")

        # ---- epochs
        def on_train_epoch_start(self, trainer, pl_module):
            ck = agent._ck_e2e
            ck.epoch = int(trainer.current_epoch)
            ck.epoch_frac = float(ck.epoch)
            ck.rank = int(getattr(trainer, "global_rank", 0))
            ck.world_size = int(getattr(trainer, "world_size", 1))
            if ck.recorder is not None:            # left over from an interrupted epoch: no DONE
                ck.recorder.flush()
                ck.recorder = None
            phase = ck.phase(ck.epoch)
            rec = None
            if ck.recording(ck.epoch) and phase != "replay":
                ck.recorder = CandRecorder(ck.cfg.io_dir, ck.rank, ck.world_size, ck.epoch, ck.cfg.rec_chunk_tokens)
                rec = ck.recorder.attempt
            stats = ck.refresh_labels() if phase == "onpolicy" else None
            if getattr(trainer, "is_global_zero", True):
                warn = label_supply_warning(ck, stats) if phase == "onpolicy" else None
                self._append("epochs.jsonl", {"event": "epoch_start", "epoch": ck.epoch, "phase": phase,
                                              "recording": rec is not None, "attempt_rank0": rec,
                                              "labels": stats, "label_warning": warn, "mb": ck.mb,
                                              "gstep": int(trainer.global_step),
                                              "world_size": ck.world_size, "time": time.time(),
                                              "ema": ck.ema.state_dict(), "cum": dict(ck.cum)})
                if warn and warn.get("warn"):
                    print("\n" + "!" * 100 + f"\n[ck_e2e] WARNING epoch {ck.epoch}: {warn['msg']}\n" + "!" * 100,
                          flush=True)

        def on_train_epoch_end(self, trainer, pl_module):
            ck = agent._ck_e2e
            if ck.recorder is not None:
                ck.recorder.close_epoch()
                n = ck.recorder.n_tokens
                ck.recorder = None
            else:
                n = None
            if getattr(trainer, "is_global_zero", True):
                self._append("epochs.jsonl", {"event": "epoch_end", "epoch": ck.epoch, "rec_tokens_rank0": n,
                                              "mb": ck.mb, "gstep": int(trainer.global_step), "time": time.time(),
                                              "skipped_steps": ck.skipped_steps, "cum": dict(ck.cum)})

        # ---- batches
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            ck = agent._ck_e2e
            nb = trainer.num_training_batches
            nb = int(nb) if nb and math.isfinite(float(nb)) else 0
            ck.epoch_frac = float(trainer.current_epoch) + (float(batch_idx) / nb if nb > 0 else 0.0)
            ck.batch_idx = int(batch_idx)
            ck.gstep = int(trainer.global_step)
            ck.rank = int(getattr(trainer, "global_rank", 0))
            self._t_start = time.time()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            ck = agent._ck_e2e
            now = time.time()
            if getattr(trainer, "is_global_zero", True) and (ck.mb % max(ck.cfg.log_every, 1) == 0):
                if self._f is None:
                    self._f = open(self._io() / "steps_rank0.jsonl", "a")
                rec = {"epoch": int(trainer.current_epoch), "batch": int(batch_idx), "gstep": int(trainer.global_step),
                       "mb": int(ck.mb), "epoch_frac": round(ck.epoch_frac, 6),
                       "sec_step": round(now - self._t_start, 4) if self._t_start else None,
                       "sec_wait": round(self._t_start - self._t_end, 4) if (self._t_end and self._t_start) else None,
                       "skipped_steps": ck.skipped_steps, "ck_clip_norm": ck.last_clip_norm}
                if torch.cuda.is_available():
                    rec["mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
                for k, v in (agent.latest_logs or {}).items():
                    if k.startswith(("ck/", "gnorm/bev_", "loss")):
                        rec[k] = float(v)
                self._f.write(json.dumps(rec) + "\n")
                self._n += 1
                if self._n % 50 == 0:
                    self._f.flush()
            every = int(ck.cfg.label_refresh_every_mb)
            if every > 0 and ck.phase(ck.epoch) == "onpolicy" and ck.mb % every == 0:
                ck.refresh_labels()
            self._t_end = now

        # ---- optimiser
        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            ck = agent._ck_e2e
            allp = [p for p in pl_module.parameters() if p.grad is not None]
            if allp:
                gn_all = torch.stack([g.float() for g in torch._foreach_norm([p.grad for p in allp])])
                if not bool(torch.isfinite(gn_all).all()):
                    for p in allp:
                        p.grad = None
                    ck.skipped_steps += 1
                    if getattr(trainer, "is_global_zero", True):
                        print(f"[ck_e2e] non-finite gradient at global_step {trainer.global_step}: optimiser step "
                              f"skipped (total {ck.skipped_steps})", flush=True)
                    return
            clip = float(ck.cfg.clip)
            params = [p for p in ck_parameters(agent) if p.grad is not None]
            if clip > 0 and params:
                ck.last_clip_norm = float(torch.nn.utils.clip_grad_norm_(params, clip))

        # ---- end
        def _close(self):
            if self._f is not None:
                self._f.close()
                self._f = None

        def on_train_end(self, trainer, pl_module):
            ck = agent._ck_e2e
            if ck.recorder is not None:
                ck.recorder.flush()
                ck.recorder = None
            self._close()
            if getattr(trainer, "is_global_zero", True):
                (self._io() / "TRAIN_DONE").write_text(json.dumps({"time": time.time(), "epoch": ck.epoch,
                                                                   "gstep": int(trainer.global_step), "mb": ck.mb}))

        def on_exception(self, trainer, pl_module, exception):
            ck = agent._ck_e2e
            try:
                if ck.recorder is not None:
                    ck.recorder.flush()
                    ck.recorder = None
            finally:
                self._close()

    return CKE2ECallback()
