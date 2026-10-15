"""CK2 e2e: v2 + CK2 student trained from scratch end to end (spec: scratchpad ck2e2e/SPEC.md; SSR/report/47 s9).

Used ONLY when ParaSSRConfig.ck_e2e2 is enabled; with the default (empty dict / enabled false) the agent never imports
this module (v2 bit-identical).  The old CK e2e (ck/online.py) is imported, never modified.

Schedule (epoch = trainer.current_epoch)
  'warmup'         epoch < record_from_epoch (4)       : per token 32 sampled raw anchors (AnchorSampler, 8 near + 8 mid
                                                         + 16 strat) + warmup.n_var (32) type-balanced rule variants
                                                         of the fixed near+mid-16 subset, official labels
                                                         (raw256 / var_separate_sampler16_accstraight); online teacher KD
                                                         on the same 64 candidates.
  'warmup_record'  record_from <= epoch < onpolicy_from : the warm-up step + the student's own v2 top-16 and its 80 rule
                                                         variants (96 per token, no grad, no teacher) are RECORDED for the
                                                         background labeler (tools/ck/e2e2/labeler2.py).
  'onpolicy'       epoch >= onpolicy_from_epoch (5)     : current top-16 + n_var_step (32) type-balanced variants of them
                                                         (48, KD / lateral KD; the 16 originals carry the surrogate) +
                                                         the label set G of generation <= e - label_lag (16 identities +
                                                         onpolicy_label.n_var (32) variants, official labels; rows without
                                                         a generation -> the warm-up-style fallback set) = 96 per token in
                                                         ONE student pass.  Recording continues until record_until.
EP target (ep_target, user 2026-10-08 ~20:10 KST; default 'official' since 2026-10-09 00:30 KST, decoupled was worse on navtest): the label targets y_lab are
  ep_target.ck_targets(official 9-column labels, ep_target) everywhere (warm-up raw / variant targets, on-policy
  generation labels, fallback set, score prior; e2e_data2); 'decoupled' = official EP without the NC * DAC * DDC factor.
  The teachers must have been trained with the same --ep-target (launch_util2.teacher_check refuses otherwise), so the
  label BCE and the EP KD target mean the same thing.  KD itself is unchanged (teacher probabilities).
KD calibration (kd_calib, user 2026-10-08 ~22:35 KST; ck/kd_calib.py): with kd_calib.enabled each teacher's score LOGITS
  are Platt-scaled per key, z' = a_k z + b_k (a_k > 0; tools/ck/e2e2/fit_kd_calib.py, NLL fit on navtrain_train r34
  top-16 candidates vs the student's label targets), BEFORE the sigmoid and the DET / MAP / mean combination
  (combine_teacher); fitted keys = the KD keys the teacher is a source of (DET nc / ep / ttc, MAP dac / ep), identity
  elsewhere.  TeacherPair2 refuses a calibration file whose run / which / ckpt sha16 / ep_target / arm differ from the
  loaded teacher or that leaves one of its KD keys unfitted.  Lateral KD is untouched.  enabled false = old path.
Student CK loss (added after the v2 grad balancer, outside it; ParaSSRAgent.compute_loss)
  L_CK2 = loss_weight * [ lambda_score * BCE(s_lab, y_lab)                       (official labels, EP per ep_target, 1.0)
                          + lambda_sur * L_sur(decode(tau_sur, z_lon = 0, w_lat; slope lon_st_slope))
                          + r * lambda_kd_score * KL(p_T || sigmoid(s_kd))       (per-key source nc/ttc <- DET,
                                                                                  dac <- MAP, ep/comfort <- mean; 0.5)
                          + r * w_lat * L1(e_lat_s, 0.5 (e_lat_DET + e_lat_MAP)) ] (ONE L1 on the mean target)
          + 0 * (score logits + w_lat of every candidate)                          (all trainable params in the graph)
  w_lat = losses.EmaBalancer(lat_kd) weight (0 before lat_kd.start_mb) or lat_kd.fixed (balance 'fixed').  The EMA is
  fed the all-rank MEANS of lambda_sur * L_sur and the lateral-KD L1 (one all-reduce per training micro-batch; the
  per-rank values are still logged as ck2/sur / ck2/lat_kd), so w_lat and the EMA state are identical on every rank
  and the rank-0 Lightning checkpoint restores every rank exactly (report 48 F4).
  No longitudinal head: trunk.lon_head is zeroed + frozen and every decode passes z_lon = 0 explicitly (student and
  teachers); trunk.gate_head is frozen.  No tau'_KD, no rescore KD, no control KD on c_lon.
  KD is off for a token without a DET or MAP BEV (combine_teacher kd_ok = det ok & map ok).
BEV-KD arm (bev_kd.enabled; ck/bev_kd_arm.py): + adapter_weight * sum_t L_t(GradScale(bev_embed, lam_t / adapter_weight))
  over bev_kd.teachers (arm: [det, map], user 2026-10-08 ~20:40 KST), one RatioController per teacher (lam_t * |dL_t/dBEV|
  ~ ratio x |g_plan|, measured per teacher on the v2 'gnorm/plan' micro-batches, all-reduced); returned inside the CK
  loss.  Tokens without the teacher's BEV (kd_ok_<i> False) are masked for that teacher and counted (bevkd_miss_<t>).
Gradient paths: candidates detached; CK -> student CK -> AdapterSGrid (x bev_grad_scale) -> v2 bev_embed; teachers
  frozen, no_grad, fp16 autocast on CUDA, plain python objects (not in state_dict / DDP / optimiser).
Non-finite: a non-finite L_CK2 is replaced by 0 * sum(student params) (counted); a non-finite gradient skips the whole
  optimiser step (callback); CK params are clipped to `clip` before Lightning's global clip.
Inference (eval, infer_outputs): ck2_* outputs over the 96-candidate pool (v2 top-16 x 6 variants, c = k * 6 + v):
  CK2 logits, the learned lateral control / offsets and every column's laterally corrected trajectory; the selection
  (ck/select2.py: old ck_final blended with v2 by beta, lateral applied after selection) happens in the evaluation.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .constants import CK_KEYS, SUR_MARGINS, SUR_TERMS, SUR_WEIGHTS
from .online import N_ROWS, _sanitize_traj, bev_sgrid, label_supply_warning, student_topk

# ----------------------------------------------------------------------------------------------- constants
# layout / stream / G_src constants and the variant-column draw, the online variants and the recorder are the data
# side's (ck/e2e_data2.py, ck/cands2.py): ONE implementation of every deterministic draw (re-exported here under the
# SPEC s1-3 names).
from . import cands2 as C2  # noqa: E402
from .cands2 import online_variants, v0_from_status  # noqa: E402,F401
from .e2e_data2 import (EP_TARGET_DEFAULT, G_K, GROUP_VAR, K16, N_NEAR_MID, NV, SRC_FALLBACK,  # noqa: E402,F401
                        SRC_NAMES, SRC_NODATA, SRC_OLDER, SRC_PREV, STREAM_LAB, STREAM_NOW, STREAM_WU,
                        TEACHER_DET_RUN, TEACHER_MAP_RUN, VAR_SAMPLING, VNAMES, CandRecorder2, draw_cols, sample_cols,
                        type_balanced_cols, uniform_cols)
from .ep_target import EP_TARGETS  # noqa: E402
from .kd_calib import calib_path  # noqa: E402

PHASES2 = ("warmup", "warmup_record", "onpolicy")
PHASE2_CODE = {p: i for i, p in enumerate(PHASES2)}
SUR_GROUP_MAX = {"near": 0, "near_mid": 1, "all": 2}
SUR_GROUP_N = {"near": 8, "near_mid": 16, "all": 32}
SCORE_PRIORS = ("ck2_warmup", "teacher", "none")
_CK = "/home/external-user/ssd/yongjae_refiner/ck"


# ----------------------------------------------------------------------------------------------- config
def _d_variants() -> Dict[str, Any]:
    return {"speeds": [-1.0, -0.5, 0.5], "lats": [-0.5, 0.5], "combine": "separate", "s_on_frac": 0.2,
            "ext": "straight", "compute_dtype": "float64"}


def _d_lat_kd() -> Dict[str, Any]:
    return {"balance": "ema", "ratio": 0.5, "m": 0.99, "floor": 1e-4, "cap": 10.0, "start_mb": 1000, "fixed": 1.0}


def _d_warmup() -> Dict[str, Any]:
    return {"packed": f"{_CK}/packed/navtrain_train", "raw_labels": f"{_CK}/labels/navtrain_train/raw256",
            "var_dir": f"{_CK}/ck2/labels/navtrain_train/var_separate_sampler16_accstraight", "sampler_seed": 0,
            "n_var": 32, "sur_groups": "near_mid", "lat_kd_groups": "near_mid"}


def _d_onpolicy_label() -> Dict[str, Any]:
    return {"n_orig": 16, "n_var": 32, "fallback": "warmup"}


def _d_kd_calib() -> Dict[str, Any]:
    # user 2026-10-08 ~22:35 KST: Platt calibration of the teacher logits for KD (files of tools/ck/e2e2/fit_kd_calib.py;
    # null = that teacher uncalibrated)
    return {"enabled": True, "det": str(calib_path(TEACHER_DET_RUN, "last")),
            "map": str(calib_path(TEACHER_MAP_RUN, "last"))}


def _d_bev_kd() -> Dict[str, Any]:
    # ratio_<t> / cap_<t>: optional per-teacher overrides of ratio / cap (None = the shared value) for that teacher's
    # RatioController (bev_kd_arm.controllers_from_cfg)
    return {"enabled": False, "teachers": ["det"], "distance": "mse", "init": "zero", "ln": True, "target_clip": None,
            "ratio": 0.1, "cap": 0.25, "m": 0.9, "w_max": None, "start_mb": 10600, "adapter_weight": 1.0,
            "lr_mult": 3.0, "weight_decay": 0.01, "ratio_det": None, "ratio_map": None, "cap_det": None,
            "cap_map": None}


BEV_KD_PER_TEACHER = ("ratio_det", "ratio_map", "cap_det", "cap_map")


_NESTED2 = {"variants": _d_variants, "lat_kd": _d_lat_kd, "warmup": _d_warmup, "onpolicy_label": _d_onpolicy_label,
            "bev_kd": _d_bev_kd, "kd_calib": _d_kd_calib}
KD_CALIB_TEACHERS = ("det", "map")
_INFER_SELECT_KEYS = ("beta", "set", "lat_mode")


def _b(v) -> bool:
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")


def _opt_float(v) -> Optional[float]:
    return None if v is None or (isinstance(v, str) and v.strip().lower() in ("none", "null", "")) else float(v)


def _opt_str(v) -> Optional[str]:
    return None if v is None or (isinstance(v, str) and v.strip().lower() in ("none", "null", "")) else str(v)


def _str_list(v) -> List[str]:
    if isinstance(v, str):
        v = [s.strip() for s in v.strip().strip("[]").split(",") if s.strip()]
    return [str(s) for s in v]


@dataclass
class CKE2E2Config:
    enabled: bool = False
    io_dir: str = ""
    topk: int = K16
    variants: Dict[str, Any] = field(default_factory=_d_variants)
    n_var_step: int = 32
    var_sampling: str = "type_balanced"
    teacher_det_run: str = TEACHER_DET_RUN           # ck2T10 (user 2026-10-09: official-EP teachers)
    teacher_map_run: str = TEACHER_MAP_RUN           # ck2M10
    teacher_which: str = "last"
    teacher_require_done: bool = True
    teacher_ep_check: bool = True                    # TeacherPair2 refuses teachers whose config.json ep_target !=
    #                                                  ep_target (false only for tests / CPU dry runs whose smoke
    #                                                  teachers predate --ep-target, i.e. 'official')
    teacher_amp: bool = True
    amp: bool = False
    bev_grad_scale: float = 1.0
    seed: int = 0
    ep_target: str = EP_TARGET_DEFAULT                # 'official' (user 2026-10-09 00:30 KST) | 'decoupled'
    score_prior: str = "ck2_warmup"
    score_prior_rows: int = 4000
    loss_weight: float = 1.0
    lambda_score: float = 1.0
    lambda_kd_score: float = 0.5
    kd_score_keys: List[str] = field(default_factory=lambda: ["nc", "dac", "ep", "ttc"])  # user 10-08: no comfort KD
    kd_calib: Dict[str, Any] = field(default_factory=_d_kd_calib)                       # user 10-08 ~22:35 KST
    lambda_sur: float = 1.0
    lat_kd: Dict[str, Any] = field(default_factory=_d_lat_kd)
    lon_st_slope: float = 0.1
    lr_mult: float = 3.0
    weight_decay: float = 0.01
    clip: float = 1.0
    warmup: Dict[str, Any] = field(default_factory=_d_warmup)
    record_from_epoch: int = 4
    record_until_epoch: Optional[int] = None
    onpolicy_from_epoch: int = 5
    kd_ramp_epochs: float = 0.0
    label_lag: int = 1
    label_refresh_every_mb: int = 1000
    onpolicy_label: Dict[str, Any] = field(default_factory=_d_onpolicy_label)
    rec_chunk_tokens: int = 64
    grad_share_every: int = 100
    log_every: int = 1
    infer_outputs: bool = True
    infer_select: Optional[Dict[str, Any]] = None
    strict_rows: float = 0.999
    bev_kd: Dict[str, Any] = field(default_factory=_d_bev_kd)

    # ------------------------------------------------------------------------------------------ parse
    @classmethod
    def from_any(cls, x: Any) -> "CKE2E2Config":
        """None / dict / DictConfig / CKE2E2Config -> CKE2E2Config (defaults merged, nested dicts merged key-wise);
        an unknown key (top level or nested) is a ValueError."""
        if isinstance(x, CKE2E2Config):
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
            raise ValueError(f"ck_e2e2 must be a mapping, got {type(x).__name__}")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(x) - known)
        if unknown:
            raise ValueError(f"ck_e2e2: unknown key(s) {unknown}")
        kw: Dict[str, Any] = {}
        for k, v in x.items():
            if k in _NESTED2:
                base = _NESTED2[k]()
                if v is None:
                    v = {}
                if not isinstance(v, dict):
                    raise ValueError(f"ck_e2e2.{k} must be a mapping, got {v!r}")
                bad = sorted(set(v) - set(base))
                if bad:
                    raise ValueError(f"ck_e2e2.{k}: unknown key(s) {bad}")
                base.update(v)
                kw[k] = base
            elif k == "infer_select":
                if v is None or (isinstance(v, str) and v.strip().lower() in ("none", "null", "")):
                    kw[k] = None
                    continue
                if not isinstance(v, dict):
                    raise ValueError(f"ck_e2e2.infer_select must be a mapping or null, got {v!r}")
                bad = sorted(set(v) - set(_INFER_SELECT_KEYS))
                if bad or set(v) != set(_INFER_SELECT_KEYS):
                    raise ValueError(f"ck_e2e2.infer_select needs exactly {list(_INFER_SELECT_KEYS)}, got {sorted(v)}")
                kw[k] = dict(v)
            else:
                kw[k] = v
        c = cls(**kw)
        c._coerce()
        return c

    def _coerce(self) -> None:
        for k in ("enabled", "teacher_require_done", "teacher_ep_check", "teacher_amp", "amp", "infer_outputs"):
            setattr(self, k, _b(getattr(self, k)))
        for k in ("topk", "n_var_step", "seed", "score_prior_rows", "record_from_epoch", "onpolicy_from_epoch",
                  "label_lag", "label_refresh_every_mb", "rec_chunk_tokens", "grad_share_every", "log_every"):
            setattr(self, k, int(getattr(self, k)))
        if self.record_until_epoch is not None:
            self.record_until_epoch = (None if str(self.record_until_epoch).strip().lower() in ("none", "null")
                                       else int(self.record_until_epoch))
        for k in ("bev_grad_scale", "loss_weight", "lambda_score", "lambda_kd_score", "lambda_sur", "lon_st_slope",
                  "lr_mult", "weight_decay", "clip", "kd_ramp_epochs", "strict_rows"):
            setattr(self, k, float(getattr(self, k)))
        ks = _str_list(self.kd_score_keys)
        if not ks or len(set(ks)) != len(ks) or any(k not in CK_KEYS for k in ks):
            raise ValueError(f"ck_e2e2.kd_score_keys must be a non-empty subset of {list(CK_KEYS)}, got {ks}")
        self.kd_score_keys = [k for k in CK_KEYS if k in ks]                  # canonical CK_KEYS order
        v = dict(self.variants)
        self.variants = {"speeds": [float(a) for a in _str_list(v["speeds"])] if isinstance(v["speeds"], str)
                         else [float(a) for a in v["speeds"]],
                         "lats": [float(a) for a in _str_list(v["lats"])] if isinstance(v["lats"], str)
                         else [float(a) for a in v["lats"]],
                         "combine": str(v["combine"]), "s_on_frac": float(v["s_on_frac"]), "ext": str(v["ext"]),
                         "compute_dtype": str(v["compute_dtype"]).replace("torch.", "")}
        e = dict(self.lat_kd)
        self.lat_kd = {"balance": str(e["balance"]), "ratio": float(e["ratio"]), "m": float(e["m"]),
                       "floor": float(e["floor"]), "cap": _opt_float(e["cap"]), "start_mb": int(e["start_mb"]),
                       "fixed": float(e["fixed"])}
        w = dict(self.warmup)
        self.warmup = {"packed": str(w["packed"]), "raw_labels": str(w["raw_labels"]), "var_dir": str(w["var_dir"]),
                       "sampler_seed": int(w["sampler_seed"]), "n_var": int(w["n_var"]),
                       "sur_groups": str(w["sur_groups"]), "lat_kd_groups": str(w["lat_kd_groups"])}
        o = dict(self.onpolicy_label)
        self.onpolicy_label = {"n_orig": int(o["n_orig"]), "n_var": int(o["n_var"]), "fallback": str(o["fallback"])}
        b = dict(self.bev_kd)
        self.bev_kd = {"enabled": _b(b["enabled"]), "teachers": _str_list(b["teachers"]), "distance": str(b["distance"]),
                       "init": str(b["init"]), "ln": _b(b["ln"]), "target_clip": _opt_float(b["target_clip"]),
                       "ratio": float(b["ratio"]), "cap": float(b["cap"]), "m": float(b["m"]),
                       "w_max": _opt_float(b["w_max"]), "start_mb": int(b["start_mb"]),
                       "adapter_weight": float(b["adapter_weight"]), "lr_mult": float(b["lr_mult"]),
                       "weight_decay": float(b["weight_decay"]),
                       **{k: _opt_float(b[k]) for k in BEV_KD_PER_TEACHER}}
        kc = dict(self.kd_calib)
        self.kd_calib = {"enabled": _b(kc["enabled"]), **{t: _opt_str(kc[t]) for t in KD_CALIB_TEACHERS}}
        if self.infer_select is not None:
            s = dict(self.infer_select)
            self.infer_select = {"beta": float(s["beta"]), "set": str(s["set"]), "lat_mode": str(s["lat_mode"])}
        self.io_dir = "" if self.io_dir is None else str(self.io_dir)
        self.teacher_det_run = "" if self.teacher_det_run in (None, "none", "None") else str(self.teacher_det_run)
        self.teacher_map_run = "" if self.teacher_map_run in (None, "none", "None") else str(self.teacher_map_run)
        self.teacher_which = str(self.teacher_which)
        self.var_sampling = str(self.var_sampling)
        self.score_prior = str(self.score_prior)
        self.ep_target = str(self.ep_target)
        if self.ep_target not in EP_TARGETS:
            raise ValueError(f"ck_e2e2.ep_target must be one of {EP_TARGETS}, got {self.ep_target!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def compute_dtype(self) -> torch.dtype:
        return {"float64": torch.float64, "float32": torch.float32}[self.variants["compute_dtype"]]

    # ------------------------------------------------------------------------------------------ validate
    def validate(self, parent=None) -> None:
        """Only when enabled.  parent: the ParaSSRConfig (old CK e2e rules + CK2 rules, SPEC s7-1)."""
        if not self.enabled:
            return
        if parent is not None:
            if getattr(parent, "refiner_mode", "off") != "off":
                raise ValueError("ck_e2e2 needs refiner_mode off (Stage E and CK2 e2e are exclusive)")
            old = getattr(parent, "ck_e2e", None) or {}
            old_on = old.get("enabled", False) if hasattr(old, "get") else getattr(old, "enabled", False)
            if _b(old_on):
                raise ValueError("ck_e2e and ck_e2e2 cannot both be enabled")
            if not getattr(parent, "plan_anchor", False):
                raise ValueError("ck_e2e2 needs plan_anchor=true (top-16 of the anchor planner)")
            if (int(parent.bev_h), int(parent.bev_w)) != (50, 100) or tuple(float(v) for v in parent.pc_range) != (
                    -32.0, 0.0, -2.0, 32.0, 32.0, 2.0):
                raise ValueError("ck_e2e2 reads the 50 x 100 S grid (0.64 m, front ROI)")
            if int(parent.embed_dims) != 256:
                raise ValueError(f"ck_e2e2 needs embed_dims 256, got {parent.embed_dims}")
        if self.topk != K16:
            raise ValueError(f"ck_e2e2.topk must be {K16} (96-column generation layout), got {self.topk}")
        v = self.variants
        try:
            C2.variants_cfg(v)        # 6-column 'separate' table in VNAMES order, ext const_curv | straight, dtype
        except ValueError as e:
            raise ValueError(f"ck_e2e2.variants: {e}") from None
        for key, n in (("n_var_step", self.n_var_step), ("warmup.n_var", self.warmup["n_var"]),
                       ("onpolicy_label.n_var", self.onpolicy_label["n_var"])):
            if not 0 <= n <= K16 * (NV - 1):
                raise ValueError(f"ck_e2e2.{key} must be in [0, {K16 * (NV - 1)}], got {n}")
        if self.onpolicy_label["n_orig"] != K16:
            raise ValueError(f"ck_e2e2.onpolicy_label.n_orig must be {K16} (all identities of the generation)")
        if self.onpolicy_label["fallback"] not in ("warmup", "none"):
            raise ValueError("ck_e2e2.onpolicy_label.fallback must be warmup | none")
        if self.var_sampling not in VAR_SAMPLING:
            raise ValueError(f"ck_e2e2.var_sampling must be one of {VAR_SAMPLING}, got {self.var_sampling!r}")
        if self.score_prior not in SCORE_PRIORS:
            raise ValueError(f"ck_e2e2.score_prior must be one of {SCORE_PRIORS}, got {self.score_prior!r}")
        for k in ("sur_groups", "lat_kd_groups"):
            if self.warmup[k] not in SUR_GROUP_MAX:
                raise ValueError(f"ck_e2e2.warmup.{k} must be one of {tuple(SUR_GROUP_MAX)}, got {self.warmup[k]!r}")
        if self.record_from_epoch > self.onpolicy_from_epoch:
            raise ValueError(f"ck_e2e2: record_from_epoch {self.record_from_epoch} > onpolicy_from_epoch "
                             f"{self.onpolicy_from_epoch} (on-policy needs recorded labels)")
        if self.label_lag < 1:
            raise ValueError(f"ck_e2e2.label_lag must be >= 1, got {self.label_lag}")
        if not self.io_dir or not os.path.isabs(self.io_dir) or "=" in self.io_dir:
            raise ValueError(f"ck_e2e2.io_dir must be an absolute path without '=', got {self.io_dir!r}")
        lk = self.lat_kd
        if lk["balance"] not in ("ema", "fixed"):
            raise ValueError(f"ck_e2e2.lat_kd.balance must be ema | fixed, got {lk['balance']!r}")
        if not (0.0 < lk["m"] < 1.0 and lk["floor"] > 0.0 and lk["ratio"] >= 0.0 and lk["fixed"] >= 0.0):
            raise ValueError(f"ck_e2e2.lat_kd needs 0 < m < 1, floor > 0, ratio >= 0, fixed >= 0, got {lk}")
        if self.rec_chunk_tokens < 1 or self.log_every < 1:
            raise ValueError("ck_e2e2.rec_chunk_tokens and log_every must be >= 1")
        if not 0.0 <= self.strict_rows <= 1.0:
            raise ValueError(f"ck_e2e2.strict_rows must be in [0, 1], got {self.strict_rows}")
        if self.infer_select is not None:
            if not 0.0 <= self.infer_select["beta"] <= 1.0:
                raise ValueError("ck_e2e2.infer_select.beta must be in [0, 1]")
        bk = self.bev_kd
        if bk["enabled"]:
            from .bev_kd import DISTANCES, INITS, TEACHERS
            if not bk["teachers"] or any(t not in TEACHERS for t in bk["teachers"]) or \
                    len(set(bk["teachers"])) != len(bk["teachers"]):
                raise ValueError(f"ck_e2e2.bev_kd.teachers must be a non-empty subset of {TEACHERS}, got "
                                 f"{bk['teachers']}")
            if bk["distance"] not in DISTANCES or bk["init"] not in INITS:
                raise ValueError(f"ck_e2e2.bev_kd distance / init must be in {DISTANCES} / {INITS}")
            if not (bk["ratio"] > 0 and bk["cap"] > 0 and 0.0 < bk["m"] < 1.0 and bk["adapter_weight"] > 0):
                raise ValueError(f"ck_e2e2.bev_kd needs ratio, cap, adapter_weight > 0 and 0 < m < 1, got {bk}")
            if bk["w_max"] is not None and not bk["w_max"] > 0:
                raise ValueError("ck_e2e2.bev_kd.w_max must be > 0 or null")
            for k in BEV_KD_PER_TEACHER:
                if bk[k] is None:
                    continue
                if not bk[k] > 0:
                    raise ValueError(f"ck_e2e2.bev_kd.{k} must be > 0 or null, got {bk[k]}")
                if k.split("_", 1)[1] not in bk["teachers"]:
                    raise ValueError(f"ck_e2e2.bev_kd.{k} is set but teacher {k.split('_', 1)[1]!r} is not in "
                                     f"bev_kd.teachers {bk['teachers']}")
        kc = self.kd_calib
        if kc["enabled"]:
            if all(kc[t] is None for t in KD_CALIB_TEACHERS):
                raise ValueError("ck_e2e2.kd_calib.enabled needs a calibration file for det and / or map")
            for t in KD_CALIB_TEACHERS:
                if kc[t] is not None and (not os.path.isabs(kc[t]) or "=" in kc[t]):
                    raise ValueError(f"ck_e2e2.kd_calib.{t} must be an absolute path without '=' or null, got "
                                     f"{kc[t]!r}")
        # teachers: both runs required (the lateral KD target is the mean of the DET and MAP offsets)
        for key, want in (("teacher_det_run", "T"), ("teacher_map_run", "M")):
            run = getattr(self, key)
            if not run:
                raise ValueError(f"ck_e2e2.{key} is required (lateral KD target = mean of DET and MAP)")
            cj = Path(run) / "config.json"
            if cj.is_file():
                arm = json.loads(cj.read_text()).get("arm")
                if arm != want:
                    raise ValueError(f"ck_e2e2.{key} {run}: config.json arm {arm!r}, expected {want!r}")
        # the warm-up variant label files must be the same variant definition (speeds, lats, combine, s_on_frac,
        # names, accel extension)
        if (Path(self.warmup["var_dir"]) / "build.json").is_file():
            try:
                C2.check_variant_files(self.warmup["var_dir"], v)
            except ValueError as e:
                raise ValueError(f"ck_e2e2.variants vs warm-up label files: {e}") from None


# ----------------------------------------------------------------------------------------------- student
def freeze_lon_gate(net) -> List[str]:
    """CK2: zero the last layer of trunk.lon_head (z_lon == 0 for every loader of the checkpoint) and freeze
    trunk.lon_head + trunk.gate_head (never in a loss; out of the optimiser and the DDP reducer).  Same rule as
    tools/ck/train_ck2.freeze_unused (re-implemented: navsim does not import tools).  -> frozen parameter names."""
    lh = net.trunk.lon_head
    with torch.no_grad():
        lh[-1].weight.zero_()
        lh[-1].bias.zero_()
    names = []
    for mod_name in ("trunk.lon_head", "trunk.gate_head"):
        mod = net.get_submodule(mod_name)
        for n, p in mod.named_parameters():
            p.requires_grad_(False)
            names.append(f"{mod_name}.{n}")
    return names


def score_prior_of(cfg: CKE2E2Config) -> Tuple[Optional[np.ndarray], str]:
    """Score-head bias prior: 'ck2_warmup' = e2e_data2.ck2_warmup_prior (expected label mean of the warm-up mix),
    'teacher' = config.json init_report.score_prior of the DET teacher run, 'none' -> (None, 'none')."""
    if cfg.score_prior == "ck2_warmup":
        from .e2e_data2 import ck2_warmup_prior
        p = ck2_warmup_prior(cfg, n_max=int(cfg.score_prior_rows))
        return (None if p is None else np.asarray(p, np.float64)), "ck2_warmup"
    if cfg.score_prior == "teacher":
        cj = Path(cfg.teacher_det_run) / "config.json"
        if cj.is_file():
            p = (json.loads(cj.read_text()).get("init_report") or {}).get("score_prior")
            if p is not None and len(p) == len(CK_KEYS):
                return np.clip(np.asarray(p, np.float64), 1e-3, 1 - 1e-3), f"teacher:{cj}"
        return None, "teacher (not found)"
    return None, "none"


def build_student_ck2(cfg: CKE2E2Config, apply_prior: bool = True):
    """CKNet('S', seed, score_hidden 256, lead_aux False), adapter.bev_grad_scale = cfg.bev_grad_scale, lon / gate
    heads frozen (lon zeroed), score-head bias = logit(prior) unless apply_prior is False (a checkpoint will be
    loaded: no file is opened then).  No global RNG is consumed (CKNet forks the RNG)."""
    from .model import CKNet

    cfg = CKE2E2Config.from_any(cfg)
    net = CKNet("S", int(cfg.seed), score_hidden=256, lead_aux=False)
    net.trunk.adapter.bev_grad_scale = float(cfg.bev_grad_scale)
    frozen = freeze_lon_gate(net)
    rep: Dict[str, Any] = {"source": None, "score_prior": None, "score_prior_source": None, "frozen": frozen}
    if apply_prior and cfg.score_prior != "none":
        prior, src = score_prior_of(cfg)
        rep["score_prior_source"] = src
        if prior is not None:
            net.set_score_prior(prior)
            rep["score_prior"] = [float(v) for v in prior]
    net.init_report = rep
    return net


# ----------------------------------------------------------------------------------------------- decode / variants
def lateral_decode(cand: torch.Tensor, w_lat: torch.Tensor, v0: torch.Tensor, slope: float) -> Dict:
    """correct(cand, z_lon = 0, w_lat, v0, slope): the learned longitudinal head is removed in CK2."""
    from .correct import correct

    w = w_lat.float()
    return correct(cand.float(), torch.zeros_like(w), w, v0, slope)


# ----------------------------------------------------------------------------------------------- teachers
class TeacherPair2:
    """Frozen CK2 DET (arm T) + MAP (arm M) teachers (ck.model.load_ck of tools/ck/train_ck2 runs), run online on the
    student's candidates.  Plain python object (not an nn.Module attribute of the agent: not in the state_dict / DDP /
    optimiser).  Load-time checks: config.json arm T / M, trainer 'train_ck2', done.json (require_done), lon head
    last layer all zero, and (ep_target given: the student's ck_e2e2.ep_target) config.json ep_target (missing =
    'official') == ep_target -- the same rule as the launch-time launch_util2.teacher_check, repeated where the
    teachers are actually loaded (report 48 S56-7).
    calib (ck_e2e2.kd_calib when enabled; None = no calibration, the old path): {'det': file | None, 'map': file |
    None} of tools/ck/e2e2/fit_kd_calib.py; each file must match its loaded teacher (kd_calib.check_calib: arm, run,
    which, ckpt sha16, ep_target, and every KD key of kd_keys this teacher is a source of fitted) or construction
    fails.  run() then uses z' = a_k z + b_k (per key) as the teacher's score logits."""

    def __init__(self, det_run: str, map_run: str, which: str = "last", device="cpu", amp: bool = True,
                 require_done: bool = True, ep_target: Optional[str] = None,
                 calib: Optional[Dict[str, Optional[str]]] = None, kd_keys: Optional[Sequence[str]] = None):
        from .ep_target import check_ep_target, run_ep_target
        from .model import load_ck

        if not det_run or not map_run:
            raise ValueError("TeacherPair2 needs both the DET and the MAP run")
        self.device = torch.device(device)
        self.runs = {"det": str(det_run), "map": str(map_run)}
        nets, eps = {}, {}
        for name, run, arm in (("det", det_run, "T"), ("map", map_run, "M")):
            rd = Path(run)
            cj = rd / "config.json"
            if not cj.is_file():
                raise ValueError(f"teacher {name} {run}: no config.json")
            cfg = json.loads(cj.read_text())
            if cfg.get("arm") != arm:
                raise ValueError(f"teacher {name} {run}: arm {cfg.get('arm')!r}, expected {arm!r}")
            if cfg.get("trainer") != "train_ck2":
                raise ValueError(f"teacher {name} {run}: trainer {cfg.get('trainer')!r} != 'train_ck2' (CK2 teacher "
                                 f"required)")
            if require_done and not (rd / "done.json").is_file():
                raise ValueError(f"teacher {name} {run}: no done.json (training not finished; "
                                 f"teacher_require_done=false only for smoke runs / tests)")
            eps[name] = got = run_ep_target(cfg)
            if ep_target is not None:
                want = check_ep_target(ep_target)
                if got != want:
                    raise ValueError(f"teacher {name} {run}: ep_target {got!r} (config.json; missing = 'official') "
                                     f"!= the student's ck_e2e2.ep_target {want!r} (label BCE and EP KD would "
                                     f"disagree)")
            net, cfg = load_ck(run, which, device="cpu")
            lw = net.trunk.lon_head[-1]
            if bool(lw.weight.detach().abs().max() > 0) or bool(lw.bias.detach().abs().max() > 0):
                raise ValueError(f"teacher {name} {run}: trunk.lon_head last layer is not all zero (not a CK2 run)")
            net.requires_grad_(False)
            net.eval()
            nets[name] = (net.to(self.device), cfg)
        self.det, self.det_cfg = nets["det"]
        self.map, self.map_cfg = nets["map"]
        self.amp = bool(amp)
        self.ep_target = eps                 # each teacher run's EP target (config.json)
        self.ckpt = {k: getattr(n, "init_report", {}).get("ckpt_sha16") for k, n in (("det", self.det),
                                                                                    ("map", self.map))}
        # KD calibration (Platt on the logits, per teacher and key); None = off (old path, nothing applied)
        self.calib: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None
        self.calib_info: Dict[str, Any] = {}
        if calib is not None:
            from .kd_calib import load_teacher_calib
            bad = sorted(set(calib) - {"det", "map"})
            if bad:
                raise ValueError(f"TeacherPair2 calib: unknown teacher(s) {bad}")
            cal = {}
            for name, arm in (("det", "T"), ("map", "M")):
                path = calib.get(name)
                if not path:
                    continue
                try:
                    a, b, info = load_teacher_calib(path, run=self.runs[name], which=which,
                                                    ckpt_sha16=self.ckpt[name], ep_target=eps[name], arm=arm,
                                                    kd_keys=kd_keys)
                except (ValueError, FileNotFoundError) as e:
                    raise ValueError(f"teacher {name} {self.runs[name]}: KD calibration refused: {e}") from None
                cal[name] = (torch.from_numpy(a).to(self.device), torch.from_numpy(b).to(self.device))
                self.calib_info[name] = info
            self.calib = cal or None

    def nets(self):
        return [self.det, self.map]

    def to(self, device) -> "TeacherPair2":
        device = torch.device(device)
        if device != self.device:
            for n in self.nets():
                n.to(device)
            if self.calib is not None:
                self.calib = {k: (a.to(device), b.to(device)) for k, (a, b) in self.calib.items()}
            self.device = device
        return self

    @torch.no_grad()
    def run(self, bev_det, ok_det, bev_map, ok_map, cand, status, cand_ok=None) -> Dict[str, torch.Tensor]:
        """bev_* [B, 256, 50, 100] (f16 cache), ok_* [B], cand [B, K, 8, 3] (finite), status [B, 8], cand_ok [B, K]
        -> kd_score_prob f32 [B,K,5] (combine_teacher: nc/ttc DET, dac MAP, ep/comfort mean), kd_e_lat f32 [B,K,6]
        = 0.5 (e_DET + e_MAP) (decode z_lon = 0, slope 0), kd_ok bool [B,K] = DET ok & MAP ok (per token) & finite
        logits / offsets of both & cand_ok; e_det / e_map [B,K,6], p_det / p_map [B,K,5] (logging).
        With calibration (self.calib) each teacher's logits are a_k z + b_k BEFORE the sigmoid / combination (p_det /
        p_map are then the calibrated probabilities); the lateral outputs are unchanged."""
        from ..refiner.e2e import ego_inputs
        from .kd import combine_teacher

        dev = self.device
        cand = cand.detach().float().to(dev)
        B, K = cand.shape[:2]
        ego = ego_inputs(status.detach().float().to(dev))
        v0 = ego[0]
        ac = torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=self.amp and dev.type == "cuda")
        logits, ws = [], []
        for net, bev in ((self.det, bev_det), (self.map, bev_map)):
            with ac:
                feat, mem = net.trunk.scene(bev.to(dev).float(), B)
                o = net.trunk.candidates(feat, mem, cand, *ego)
                s = net.score(o, B, K)
            logits.append(s.float())
            ws.append(o["w_lat"].float())
        if self.calib is not None:            # KD calibration: Platt on the logits, per teacher and key
            for i, name in enumerate(("det", "map")):
                ab = self.calib.get(name)
                if ab is not None:
                    logits[i] = logits[i] * ab[0] + ab[1]
        # ONE batched decode for both teachers (correct() is launch-bound; candidates decode independently)
        dec = lateral_decode(torch.cat([cand, cand], 1), torch.cat(ws, 1), v0, 0.0)
        e = dec["e_lat"][..., 2:].float()
        e_det, e_map = e[:, :K], e[:, K:]

        def tok_ok(ok_tok, s, ee):
            o = ok_tok.reshape(-1).bool().to(dev)[:, None].expand(-1, K)
            return o & torch.isfinite(s).all(-1) & torch.isfinite(ee).all(-1)

        det = {"score_logit": logits[0], "ok": tok_ok(ok_det, logits[0], e_det)}
        mp = {"score_logit": logits[1], "ok": tok_ok(ok_map, logits[1], e_map)}
        comb = combine_teacher(det, mp)
        kd_ok = comb["kd_ok"].bool()
        if cand_ok is not None:
            kd_ok = kd_ok & cand_ok.bool().to(dev)
        e_mean = 0.5 * (e_det + e_map)
        return {"kd_score_prob": comb["kd_score_prob"].float(), "kd_e_lat": e_mean, "kd_ok": kd_ok,
                "e_det": e_det, "e_map": e_map, "p_det": torch.sigmoid(logits[0]), "p_map": torch.sigmoid(logits[1])}


# ----------------------------------------------------------------------------------------------- student forward
def student_forward2(student, bev_grid: torch.Tensor, cand: torch.Tensor, status: torch.Tensor,
                     extra: Optional[torch.Tensor] = None) -> Dict:
    """One trunk.scene + ONE trunk.candidates over cat(cand, extra) (candidates are processed independently) ->
    score_logit [B,K,5] f32, extra_score_logit [B,K2,5] | None, w_lat [B,K,6], w_lat_extra [B,K2,6] | None,
    score_all / w_lat_all (every candidate), ego.  No decode here (callers use lateral_decode, z_lon = 0)."""
    from ..refiner.e2e import ego_inputs

    B, K = cand.shape[:2]
    ego = ego_inputs(status)
    v0, a0, eds, cmd = ego
    allc = cand.float() if extra is None else torch.cat([cand.float(), extra.float()], 1)
    KA = allc.shape[1]
    feat, mem = student.trunk.scene(bev_grid, B)
    o = student.trunk.candidates(feat, mem, allc, v0, a0, eds, cmd)
    sc = student.score(o, B, KA)
    w = o["w_lat"].float()
    return {"score_logit": sc[:, :K], "extra_score_logit": None if extra is None else sc[:, K:],
            "w_lat": w[:, :K], "w_lat_extra": None if extra is None else w[:, K:], "score_all": sc, "w_lat_all": w,
            "ego": ego}


# ----------------------------------------------------------------------------------------------- helpers
def _f(x) -> float:
    return float(x.detach()) if torch.is_tensor(x) else float(x)


def _masked_l1(es: torch.Tensor, et: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """mean over the m candidates and the 6 controls of |es - et| (et detached; 0 with the graph kept if m is empty)."""
    et = et.detach().float().to(es.device)
    m = m.bool().to(es.device) & torch.isfinite(et).all(-1)
    et = torch.where(m[..., None], et, es.detach().float())
    per = (es.float() - et).abs().mean(-1)
    mf = m.float()
    return (per * mf).sum() / torch.clamp(mf.sum(), min=1.0)


def _gather_c(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x [B, C, ...], idx [B, n] (long, same device) -> x[b, idx[b]] [B, n, ...]."""
    shape = idx.shape + x.shape[2:]
    ii = idx.reshape(idx.shape + (1,) * (x.dim() - 2)).expand(shape)
    return x.gather(1, ii)


def _dist_mean_many(xs: Sequence[float], device) -> List[float]:
    """element-wise mean of a float vector over the ranks (ONE all-reduce, float64) when torch.distributed is
    initialised (every rank must call it with the same length); the input unchanged otherwise."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return [float(x) for x in xs]
    dev = torch.device(device)
    if dist.get_backend() == "nccl" and dev.type != "cuda":
        dev = torch.device("cuda", torch.cuda.current_device())
    t = torch.tensor([float(x) for x in xs], dtype=torch.float64, device=dev)
    dist.all_reduce(t)
    return [float(v) / dist.get_world_size() for v in t.tolist()]


# ----------------------------------------------------------------------------------------------- main helper
class CKE2E2:
    """Loss / inference helper held by ParaSSRAgent (plain object).  epoch / epoch_frac / rank / world_size / gstep /
    batch_idx are set by the callback (make_ck2_callback); mb counts this rank's micro-batches."""

    def __init__(self, cfg: CKE2E2Config, max_epochs: int = 30):
        from .losses import EmaBalancer

        self.cfg = CKE2E2Config.from_any(cfg)
        self.max_epochs = int(max_epochs)
        e = self.cfg.lat_kd
        self.ema = EmaBalancer(ratio=e["ratio"], m=e["m"], floor=e["floor"], cap=e["cap"], start_step=e["start_mb"])
        self.bev_ctrl = None                 # bev_kd_arm.TeacherRatioControllers (one RatioController per teacher)
        if self.cfg.bev_kd["enabled"]:
            from .bev_kd_arm import controllers_from_cfg
            self.bev_ctrl = controllers_from_cfg(self.cfg.bev_kd)
        self.epoch, self.epoch_frac, self.rank, self.world_size, self.batch_idx, self.gstep = 0, 0.0, 0, 1, 0, 0
        self.mb = 0
        self.skipped_steps = 0
        self.cum = {"tokens": 0, "rows_ok": 0, "nonfinite_loss": 0, "sur_nonfinite": 0, "rec_tokens": 0,
                    "bevkd_nonfinite": 0, **{f"G_src_{n}": 0 for n in SRC_NAMES}}
        if self.cfg.bev_kd["enabled"]:       # rank-local token counters of the arm (tokens seen / without teacher BEV)
            self.cum.update({"bevkd_tok": 0, **{f"bevkd_miss_{t}": 0 for t in self.cfg.bev_kd["teachers"]}})
        self._teachers: Optional[TeacherPair2] = None
        self._labels = None
        self._labels_max_epoch: Optional[int] = None
        self._sampler = None
        self._rowmap = None
        self.recorder: Optional[CandRecorder2] = None
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
            return "warmup_record"
        return "warmup"

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
                "skipped_steps": int(self.skipped_steps),
                "bev_ctrl": None if self.bev_ctrl is None else self.bev_ctrl.state_dict()}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        if not sd:
            return
        self.ema.load_state_dict(sd["ema"])
        self.mb = int(sd.get("mb", 0))
        self.cum.update({k: int(v) for k, v in dict(sd.get("cum", {})).items()})
        self.skipped_steps = int(sd.get("skipped_steps", 0))
        if self.bev_ctrl is not None and sd.get("bev_ctrl"):
            self.bev_ctrl.load_state_dict(sd["bev_ctrl"])

    # -------------------------------------------------------------------------------------- lazy pieces
    def teachers(self, device) -> TeacherPair2:
        if self._teachers is None:
            c = self.cfg
            kc = c.kd_calib
            self._teachers = TeacherPair2(c.teacher_det_run, c.teacher_map_run, c.teacher_which, device,
                                          c.teacher_amp, c.teacher_require_done,
                                          ep_target=c.ep_target if c.teacher_ep_check else None,
                                          calib=({"det": kc["det"], "map": kc["map"]} if kc["enabled"] else None),
                                          kd_keys=c.kd_score_keys)
            if self._teachers.calib_info:
                print("[ck_e2e2] KD calibration: " + "; ".join(
                    f"{t} {i['path']} (sha16 {i['sha16']}, ckpt {i['ckpt_sha16']}, keys {i['fitted_keys']})"
                    for t, i in self._teachers.calib_info.items()), flush=True)
        return self._teachers.to(device)

    def labels(self):
        if self._labels is None:
            from .e2e_data2 import LabelStore2
            self._labels = LabelStore2(self.cfg.io_dir, n_rows=N_ROWS, packed=self.cfg.warmup["packed"],
                                       var_sampling=self.cfg.var_sampling, ep_target=self.cfg.ep_target)
        return self._labels

    def sampler(self):
        if self._sampler is None:
            from .anchor_sampler import load_anchors
            from .e2e_data2 import WarmupSampler
            w = self.cfg.warmup
            self._sampler = WarmupSampler(load_anchors(), sampler_seed=w["sampler_seed"], seed=self.cfg.seed,
                                          n_var=w["n_var"], var_sampling=self.cfg.var_sampling)
        return self._sampler

    def rowmap(self):
        if self._rowmap is None:
            from .e2e_data import RowMap
            self._rowmap = RowMap(self.cfg.warmup["packed"])
        return self._rowmap

    def tokens_of(self, rows) -> List[str]:
        r = rows.detach().cpu().numpy() if torch.is_tensor(rows) else np.asarray(rows)
        r = r.astype(np.int64).reshape(-1)
        if not (r >= 0).any():
            return [""] * len(r)
        tok = self.rowmap().tokens
        return [str(tok[x]) if 0 <= x < len(tok) else "" for x in r]

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
            raise RuntimeError("ck_e2e2: no training token of the first micro-batch has a packed navtrain_train row "
                               "(targets['ck_row'] all -1): wrong split / token mapping")
        if not self._rows_checked and self.cum["tokens"] >= 2000:
            self._rows_checked = True
            frac = self.cum["rows_ok"] / max(self.cum["tokens"], 1)
            if frac < self.cfg.strict_rows:
                raise RuntimeError(f"ck_e2e2: only {frac:.4f} of the first {self.cum['tokens']} training tokens have "
                                   f"a packed row (< strict_rows {self.cfg.strict_rows})")

    # -------------------------------------------------------------------------------------- step candidates
    @staticmethod
    def _group_pos(group_a: torch.Tensor, gmax: int, n: int) -> torch.Tensor:
        """slot positions (slot order) of the anchors with group <= gmax: exactly n per token (WarmupSampler gives
        the canonical 8 / 8 / 16 group pattern to rows without data)."""
        return torch.sort((group_a > gmax).to(torch.uint8), dim=1, stable=True).indices[:, :n]

    def warmup_batch(self, targets, toks: Sequence[str], fallback: torch.Tensor, device) -> Dict[str, Any]:
        """SPEC s2-1 (cands2.warmup_cands): 32 sampled raw anchors + warmup.n_var variants with official labels (as CK
        targets with the EP of cfg.ep_target, made by the target builder's wu_row);
        candidates that are not teacher-ok (no warm-up data, invalid / padding variant) are replaced by the v2
        trajectory (finite input; they are masked everywhere)."""
        c = self.cfg
        dev = torch.device(device)
        W = C2.warmup_cands(targets, toks, self.epoch, self.sampler(), device=dev)
        cand, fin = _sanitize_traj(W["cand"], fallback, W["teacher_ok"])
        grp = W["group"]
        ga = grp[:, :W["anc_idx"].shape[1]]
        gs = c.warmup["sur_groups"]
        sur_pos = self._group_pos(ga, SUR_GROUP_MAX[gs], SUR_GROUP_N[gs])
        lat_mask = (grp <= SUR_GROUP_MAX[c.warmup["lat_kd_groups"]]) | (grp == GROUP_VAR)
        wu = W["wu_ok"].bool()
        B = cand.shape[0]
        vv = targets["ck2_var_valid"].reshape(B, K16, NV)
        return {"cand": cand, "cand_ok": W["teacher_ok"] & fin, "extra": None, "y": W["y"],
                "ok_lab": W["ok"] & fin, "vtype_lab": W["vtype"], "grp": grp, "sur_pos": sur_pos,
                "sur_tok_ok": wu, "lat_mask": lat_mask, "kd_tok_ok": wu, "wu_ok": wu,
                "var_valid_frac": float(vv[:, :, 1:].float().mean()),
                "src": torch.full((B,), SRC_FALLBACK, dtype=torch.long)}

    def label_set(self, rows: torch.Tensor, toks: Sequence[str], targets, device) -> Dict[str, torch.Tensor]:
        """The on-policy BCE set G: LabelStore2.lookup (newest generation <= e - label_lag; 16 identities +
        onpolicy_label.n_var variants, stream 'ck2e2e.lab') + the warm-up-style fallback (cands2.fill_fallback, NU12)
        for rows without a generation (onpolicy_label.fallback 'none': those rows get G_ok False, G_src 3)."""
        c = self.cfg
        if self._labels_max_epoch != self.epoch - c.label_lag:
            self.refresh_labels()
        ol = c.onpolicy_label
        G = self.labels().lookup(rows.detach().cpu().long(), list(toks), int(self.epoch), int(c.seed),
                                 n_orig=K16, n_var=ol["n_var"])
        if ol["fallback"] == "warmup":
            G = C2.fill_fallback(G, targets, list(toks), int(self.epoch), self.sampler(), ol["n_var"])
        else:
            miss = ~G["has"].bool()
            G["G_ok"][miss] = False
            G["G_src"][miss] = SRC_NODATA
        return {k: v.to(device) for k, v in G.items()}

    def current_batch(self, predictions, v0v: torch.Tensor, rows: torch.Tensor, toks: Sequence[str], targets,
                      fallback: torch.Tensor) -> Dict[str, Any]:
        """SPEC s2-2 (cands2.onpolicy_cands): current top-16 + n_var_step variants of them (KD / lateral KD; stream
        'ck2e2e.now') + the label set G (BCE).  v0v = cands2.v0_from_status (float64, as the variant label files)."""
        c = self.cfg
        dev = fallback.device
        top = student_topk(predictions, K16)
        V = C2.online_variants(top["cand"], v0v, c.variants)
        P = C2.onpolicy_cands(top["cand"], v0v, list(toks), int(self.epoch), int(c.seed), c.n_var_step, c.variants,
                              c.var_sampling, STREAM_NOW, V=V)
        cand, fin = _sanitize_traj(P["cand"], fallback, P["ok"])
        G = self.label_set(rows, toks, targets, dev)
        G_traj, G_fin = _sanitize_traj(G["G_traj"].float(), fallback, G["G_ok"].bool())
        B, Kc = cand.shape[:2]
        return {"cand": cand, "cand_ok": P["ok"] & fin, "extra": G_traj, "y": G["G_y"].float(),
                "ok_lab": G["G_ok"].bool() & G_fin, "vtype_lab": G["G_vtype"].long(),
                "sur_pos": torch.arange(K16, device=dev)[None].expand(B, -1),
                "sur_tok_ok": torch.ones(B, dtype=torch.bool, device=dev),
                "lat_mask": torch.ones(B, Kc, dtype=torch.bool, device=dev),
                "kd_tok_ok": torch.ones(B, dtype=torch.bool, device=dev), "src": G["G_src"].long(),
                "var_valid_frac": float(V["valid96"].reshape(B, K16, NV)[:, :, 1:].float().mean()),
                "top": top, "V": V}

    # -------------------------------------------------------------------------------------- inference
    @torch.no_grad()
    def infer(self, student, features, predictions) -> Dict[str, torch.Tensor]:
        """eval-mode outputs over the 96-candidate pool (SPEC s5-1): fp32, decode slope 0, z_lon = 0.  Columns
        c = k * 6 + v (k = v2 top-16 rank, v = VNAMES); ck2_cand96[:, ::6] = top-16, [:, 0] = v2's trajectory.
        predictions['trajectory'] is untouched (v2's, NU30)."""
        from ..refiner.e2e import ego_inputs

        bev = predictions["bev_embed"]
        dev = bev.device
        top = student_topk(predictions, K16)
        status = features["status_feature"].float().to(dev)
        v0 = ego_inputs(status)[0]
        V = C2.online_variants(top["cand"], v0_from_status(status), self.cfg.variants)
        cand, fin = _sanitize_traj(V["traj96"], top["cand"][:, 0])
        with torch.autocast(device_type=dev.type, enabled=False):
            out = student_forward2(student, bev_sgrid(bev.float()), cand, status)
        lat = lateral_decode(cand, out["w_lat"], v0, 0.0)
        res = {"ck2_cand_idx": top["idx"], "ck2_v2_final": top["final"], "ck2_v2_im": top["im"],
               "ck2_v2_sim": top["sim"], "ck2_cand96": V["traj96"], "ck2_valid96": V["valid96"] & fin,
               "ck2_score_logit": out["score_logit"].float(), "ck2_w_lat": out["w_lat"].float(),
               "ck2_e_lat": lat["e_lat"][..., 2:].float(), "ck2_lat_traj": lat["traj"].float()}
        sel = self.cfg.infer_select
        if sel is not None:
            from . import select2 as S2
            idx = S2.select96(res["ck2_v2_final"], res["ck2_v2_im"], torch.sigmoid(res["ck2_score_logit"]),
                              res["ck2_valid96"], sel["beta"], types=S2.VARIANT_SETS[sel["set"]])
            idx_t = torch.as_tensor(np.asarray(idx.cpu() if torch.is_tensor(idx) else idx), dtype=torch.long,
                                    device=dev)
            traj, _src = S2.final_traj(res["ck2_cand96"], res["ck2_lat_traj"], idx_t, sel["lat_mode"])
            res["ck2_sel_idx"] = idx_t
            res["ck2_traj"] = torch.as_tensor(np.asarray(traj.cpu() if torch.is_tensor(traj) else traj),
                                              dtype=torch.float32, device=dev)
        return res

    # -------------------------------------------------------------------------------------- training loss
    def loss(self, student, features, targets, predictions, v2_loss: Optional[torch.Tensor] = None,
             v2_logs: Optional[Dict] = None, bev_kd=None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        from ..refiner.e2e import ego_inputs
        from . import losses as L

        t0 = time.time()
        c = self.cfg
        bev = predictions["bev_embed"]
        dev = bev.device
        B = bev.shape[0]
        phase = self.phase(self.epoch)
        r = self.r(self.epoch_frac)
        status = features["status_feature"].float().to(dev)
        ego = ego_inputs(status)
        v0, a0 = ego[0], ego[1]
        rows = (targets["ck_row"].reshape(-1).long() if "ck_row" in targets
                else torch.full((B,), -1, dtype=torch.long, device=dev))
        self._check_rows(rows)
        toks = self.tokens_of(rows)
        human = targets["trajectory"].float().to(dev)
        fallback = student_topk(predictions, 1)["cand"][:, 0]
        logs: Dict[str, float] = {}

        # ---- candidates of this step
        tv = time.time()
        v0v = v0_from_status(status)                    # float64 hypot (= the variant label files' v0)
        rec = None
        if phase == "onpolicy":
            S = self.current_batch(predictions, v0v, rows, toks, targets, fallback)
            if self.recording(self.epoch):
                rec = (S["top"], S["V"])
        else:
            S = self.warmup_batch(targets, toks, fallback, dev)
            if phase == "warmup_record" and self.recording(self.epoch):
                top = student_topk(predictions, K16)
                rec = (top, C2.online_variants(top["cand"], v0v, c.variants))
        t_var = time.time() - tv
        n_rec = 0
        if rec is not None and self.recorder is not None:
            top, V = rec
            n_rec = self.recorder.add(rows, V["traj96"], V["valid96"], top["idx"], top["final"], v0v, self.gstep)
            self.cum["rec_tokens"] += n_rec

        # ---- teachers (online, every phase) on the step's KD candidates
        tt = time.time()
        tp = self.teachers(dev)
        T = tp.run(targets["kd_bev_0"], targets["kd_ok_0"], targets["kd_bev_1"], targets["kd_ok_1"], S["cand"], status,
                   cand_ok=S["cand_ok"])
        t_teacher = time.time() - tt

        # ---- student forward (one pass over KD candidates + label set)
        ts = time.time()
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=bool(c.amp and dev.type == "cuda")):
            out = student_forward2(student, bev_sgrid(bev), S["cand"], status, extra=S["extra"])
        s_kd = out["score_logit"].float()
        s_lab = s_kd if S["extra"] is None else out["extra_score_logit"].float()
        corr = lateral_decode(S["cand"], out["w_lat"], v0, c.lon_st_slope)
        e_s = corr["e_lat"][..., 2:].float()
        Kc = S["cand"].shape[1]

        # ---- loss
        l_bce, per = L.score_bce(s_lab, S["y"], S["ok_lab"])
        inner = c.lambda_score * l_bce
        ref_gt_ok = (targets["ref_gt_ok"].reshape(-1).bool().to(dev) if "ref_gt_ok" in targets
                     else torch.zeros(B, dtype=torch.bool, device=dev))
        idx = L.surrogate_index(ref_gt_ok & S["sur_tok_ok"], human)
        n = int(idx.numel())
        n_s = int(S["sur_pos"].shape[1])
        l_sur = bev.new_zeros((), dtype=torch.float32)
        terms = {k: 0.0 for k in SUR_TERMS + ("P1_minus_P0",)}
        nonfin = 0
        if n > 0 and c.lambda_sur:
            ref = {k: v.to(dev) for k, v in targets.items() if k.startswith("ref_")}
            tau_sur = _gather_c(S["cand"], S["sur_pos"])
            sb = L.surrogate_batch_k(ref, idx, tau_sur, human, v0, a0)
            drows = (idx[:, None] * Kc + S["sur_pos"][idx]).reshape(-1)
            dec = L.restrict_dec(corr["raw"], drows)
            l_sur, terms, nonfin = L.correction_loss(dec, sb, n, n_s, SUR_WEIGHTS, SUR_MARGINS)
            inner = inner + c.lambda_sur * l_sur
        l_sur_val = _f(l_sur)
        kd_ok = T["kd_ok"].bool() & S["kd_tok_ok"][:, None]
        ki = [CK_KEYS.index(k) for k in c.kd_score_keys]      # user 2026-10-08: comfort excluded from score KD
        l_ks = L.score_kd_bce(s_kd[..., ki], T["kd_score_prob"][..., ki], kd_ok)
        inner = inner + (r * c.lambda_kd_score) * l_ks
        m_lat = kd_ok & S["lat_mask"]
        l_lat = _masked_l1(e_s, T["kd_e_lat"], m_lat)
        l_lat_val = _f(l_lat)
        lk = c.lat_kd
        if lk["balance"] == "fixed":
            w = float(lk["fixed"]) if self.mb >= lk["start_mb"] else 0.0
        else:
            # EMA inputs = the all-rank means of this micro-batch (ONE 2-float all-reduce, reached by every rank on
            # every training mb before the conditional BEV-KD collective): the EMA state is identical on every rank,
            # so the rank-0 checkpoint restores all ranks exactly (report 48 F4); single process: unchanged values
            ema_sur, ema_kd = _dist_mean_many([c.lambda_sur * l_sur_val, l_lat_val], dev)
            self.ema.update(ema_sur, ema_kd)
            w = float(self.ema.weight(self.mb))
        inner = inner + (r * w) * l_lat
        loss = c.loss_weight * inner
        loss = loss + 0.0 * (out["score_all"].sum() + out["w_lat_all"].sum())
        loss_val = _f(loss)
        bad_loss = not math.isfinite(loss_val)
        if bad_loss:
            self.cum["nonfinite_loss"] += 1
            loss = 0.0 * sum(p.sum() for p in student.parameters() if p.requires_grad)
        t_student = time.time() - ts

        # ---- BEV-KD arm (outside the balancer; returned inside the CK loss)
        term, term_t = None, {}
        if bev_kd is not None:
            term, term_t = self._bev_kd_term(bev_kd, bev, targets, v2_logs, logs)

        # ---- grad share (every grad_share_every micro-batches)
        every = int(c.grad_share_every)
        if (every > 0 and self.mb % every == 0 and v2_loss is not None and bev.requires_grad
                and torch.is_grad_enabled() and not bad_loss):
            parts = (("v2", v2_loss), ("ck", loss)) + ((("bevkd", term),) if term is not None else ())
            gn = {}
            for k, tl in parts:
                gg = (torch.autograd.grad(tl, bev, retain_graph=True, allow_unused=True)[0]
                      if isinstance(tl, torch.Tensor) and tl.requires_grad else None)
                gn[k] = 0.0 if gg is None else float(gg.float().norm())
                logs[f"gnorm/bev_{k}"] = gn[k]
            logs["ck2/gshare"] = gn["ck"] / max(sum(gn.values()), 1e-12)
            for t, tt in term_t.items():          # per-teacher weighted BEV-KD gradient (= lam_t |dL_t / dBEV|)
                gg = (torch.autograd.grad(tt, bev, retain_graph=True, allow_unused=True)[0]
                      if isinstance(tt, torch.Tensor) and tt.requires_grad else None)
                logs[f"gnorm/bev_bevkd_{t}"] = 0.0 if gg is None else float(gg.float().norm())

        # ---- logs
        lw = c.loss_weight
        logs.update({
            "ck2/loss": loss_val if not bad_loss else 0.0, "ck2/loss_nonfinite": float(bad_loss),
            "ck2/bce": _f(l_bce), "ck2/kd_score": _f(l_ks), "ck2/sur": l_sur_val, "ck2/sur_nonfinite": float(nonfin),
            "ck2/lat_kd": l_lat_val, "ck2/w_lat": w, "ck2/r": r, "ck2/phase": float(PHASE2_CODE[phase]),
            "ck2/mb": float(self.mb), "ck2/n_gt": float(n),
            "ck2/wt_bce": lw * c.lambda_score * _f(l_bce), "ck2/wt_sur": lw * c.lambda_sur * l_sur_val,
            "ck2/wt_kd_score": lw * r * c.lambda_kd_score * _f(l_ks), "ck2/wt_lat_kd": lw * r * w * l_lat_val,
        })
        for k, v in (per or {}).items():
            logs[f"ck2/bce_{k}"] = float(v)
        for k, v in (terms or {}).items():
            logs[f"ck2/t_{k}"] = float(v)
        if v2_loss is not None:
            logs["ck2/ratio_v2"] = (loss_val / max(abs(_f(v2_loss)), 1e-12)) if not bad_loss else 0.0
        with torch.no_grad():
            vt = S["vtype_lab"].to(dev)
            okl = S["ok_lab"]
            for name, m in (("orig", vt == 0), ("var", vt > 0)):
                lb, _ = L.score_bce(s_lab.detach(), S["y"], okl & m)
                logs[f"ck2/bce_{name}"] = float(lb)
            logs["ck2/n_var_ok"] = float((okl & (vt > 0)).sum())
            logs["ck2/G_ok_frac"] = float(okl.float().mean()) if okl.numel() else 0.0
            logs["ck2/kd_ok_frac"] = float(kd_ok.float().mean())
            logs["ck2/var_valid_frac"] = float(S["var_valid_frac"])
            srcn = S["src"].reshape(-1).cpu()
            if phase == "onpolicy":
                for code, name in enumerate(SRC_NAMES):
                    cnt = int((srcn == code).sum())
                    logs[f"ck2/G_src_{name}"] = cnt / max(B, 1)
                    self.cum[f"G_src_{name}"] += cnt
            mf = m_lat.float()
            den = float(mf.sum().clamp(min=1.0))
            logs["ck2/e_abs"] = float((e_s.detach().abs().mean(-1) * mf).sum()) / den
            logs["ck2/e_abs_T"] = float((T["kd_e_lat"].abs().mean(-1).nan_to_num(0.0) * mf).sum()) / den
            logs["ck2/e_abs_det"] = float((T["e_det"].abs().mean(-1).nan_to_num(0.0) * mf).sum()) / den
            logs["ck2/e_abs_map"] = float((T["e_map"].abs().mean(-1).nan_to_num(0.0) * mf).sum()) / den
            logs["ck2/lat_disagree"] = float(((T["e_det"] - T["e_map"]).abs().mean(-1).nan_to_num(0.0)
                                              * mf).sum()) / den
            logs["ck2/n_cand"] = float(Kc + (0 if S["extra"] is None else S["extra"].shape[1]))
        self.cum["sur_nonfinite"] += int(nonfin)
        logs["ck2/variants_ms"] = 1e3 * t_var
        logs["ck2/teacher_ms"] = 1e3 * t_teacher
        logs["ck2/student_ms"] = 1e3 * t_student
        logs["ck2/rec_tokens"] = float(n_rec)
        logs["ck2/total_ms"] = 1e3 * (time.time() - t0)
        if term is not None:
            loss = loss + term
        self.mb += 1
        return loss, {k: torch.tensor(float(v), device=dev) for k, v in logs.items()}

    def _bev_kd_term(self, bev_kd, bev: torch.Tensor, targets, v2_logs: Optional[Dict], logs: Dict[str, float]
                     ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """-> (term, {t: term_t}): term = sum_t term_t, term_t = adapter_weight * L_t(GradScale(bev_embed, lam_t /
        adapter_weight)) (BEV gradient lam_t * dL_t / d bev_embed), lam_t = teacher t's own RatioController weight of
        this mb.  On the v2 'gnorm/plan' micro-batches the unit BEV gradient norm of every teacher's loss is measured
        (one all-reduce of the per-teacher vector, mean over the ranks) and each controller is updated against the
        same g_plan (the new lam_t apply from the next mb).  teachers [det] = the old single-controller arm exactly.
        A non-finite term is replaced by 0 * (adapter parameters) and counted ({} per-teacher terms then)."""
        teachers = tuple(bev_kd.teachers)
        ctrl = self.bev_ctrl
        if ctrl is None:
            from .bev_kd_arm import controllers_from_cfg
            ctrl = self.bev_ctrl = controllers_from_cfg(self.cfg.bev_kd, teachers)
        if tuple(ctrl.teachers) != teachers:
            raise ValueError(f"BEV-KD arm teachers {list(teachers)} != controller teachers {list(ctrl.teachers)}")
        lams = ctrl.weights(self.mb)
        bo = bev_kd(bev, targets, lams)
        term = bo["term"]
        term_t = {t: bo[f"term/{t}"] for t in teachers}
        bad = not math.isfinite(_f(term))
        if bad:
            self.cum["bevkd_nonfinite"] += 1
            term = 0.0 * sum(p.sum() for p in bev_kd.parameters() if p.requires_grad)
            term_t = {}
        measure = (v2_logs is not None and "gnorm/plan" in v2_logs and bev.requires_grad and torch.is_grad_enabled())
        if measure:
            g_loc = bev_kd.unit_grad_norms(bo, teachers) if not bad else {t: float("nan") for t in teachers}
            g_all = _dist_mean_many([g_loc[t] for t in teachers], bev.device)   # every rank reaches it on the same mb
            g_unit = dict(zip(teachers, g_all))
            g_plan = _f(v2_logs["gnorm/plan"])
            ctrl.update(g_unit, g_plan)
            logs["bevkd/g_plan"] = g_plan
            for t in teachers:
                g = g_unit[t]
                logs[f"bevkd/{t}/g_kd_unit"] = g
                logs[f"bevkd/{t}/ratio_now"] = (lams[t] * g / max(g_plan, 1e-30) if math.isfinite(g)
                                                else float("nan"))
        B = int(bev.shape[0])
        self.cum["bevkd_tok"] = self.cum.get("bevkd_tok", 0) + B
        for t in teachers:
            logs[f"bevkd/{t}/loss"] = _f(bo[f"raw/{t}"])
            logs[f"bevkd/{t}/lam"] = float(lams[t])
            logs[f"bevkd/{t}/ok_frac"] = _f(bo[f"ok_frac/{t}"])
            for k in ("cos", "fve", "w_dev"):
                logs[f"bevkd/{t}/{k}"] = _f(bo[f"{k}/{t}"])
            miss = B - int(bo[f"n_ok/{t}"])
            self.cum[f"bevkd_miss_{t}"] = self.cum.get(f"bevkd_miss_{t}", 0) + miss
        logs["bevkd/term"] = _f(term)
        logs["bevkd/nonfinite"] = float(bad)
        return term, term_t


# ----------------------------------------------------------------------------------------------- callback
def ck2_parameters(agent) -> List[torch.nn.Parameter]:
    """Trainable student CK parameters (lon / gate heads have requires_grad False and are excluded)."""
    return [p for p in agent.ck_student.parameters() if p.requires_grad]


STEP_LOG_PREFIXES = ("ck2/", "bevkd/", "gnorm/", "loss")


def make_ck2_callback(agent):
    import pytorch_lightning as pl

    class CKE2E2Callback(pl.Callback):
        """Schedule (epoch, epoch_frac, rank), CandRecorder2 open / close, LabelStore2 refresh, the CK-only gradient
        clip and the non-finite step skip (before Lightning's global clip), rank-0 steps_rank0.jsonl / epochs.jsonl,
        and the CKE2E2 state (lateral-KD EMA, BEV-KD controller, mb, counters) in the Lightning checkpoint.  Lightning
        writes rank 0's state and loads it on every rank: complete for the EMA and the BEV-KD controllers (both fed
        all-reduced inputs, identical on every rank); the counters in 'cum' are rank-0 diagnostics."""

        def __init__(self):
            super().__init__()
            self._f = None
            self._n = 0
            self._t_start = self._t_end = None

        @property
        def state_key(self) -> str:
            return "CKE2E2Callback"

        def state_dict(self):
            return agent._ck_e2e2.state_dict()

        def load_state_dict(self, state_dict):
            agent._ck_e2e2.load_state_dict(state_dict)

        # ---- helpers
        def _io(self) -> Path:
            p = Path(agent._ck_e2e2.cfg.io_dir)
            p.mkdir(parents=True, exist_ok=True)
            return p

        def _append(self, name: str, rec: Dict) -> None:
            with open(self._io() / name, "a") as f:
                f.write(json.dumps(rec) + "\n")

        # ---- epochs
        def on_train_epoch_start(self, trainer, pl_module):
            ck = agent._ck_e2e2
            ck.epoch = int(trainer.current_epoch)
            ck.epoch_frac = float(ck.epoch)
            ck.rank = int(getattr(trainer, "global_rank", 0))
            ck.world_size = int(getattr(trainer, "world_size", 1))
            if ck.recorder is not None:            # left over from an interrupted epoch: no DONE
                ck.recorder.flush()
                ck.recorder = None
            phase = ck.phase(ck.epoch)
            rec = None
            if ck.recording(ck.epoch) and phase != "warmup":
                ck.recorder = CandRecorder2(ck.cfg.io_dir, ck.rank, ck.world_size, ck.epoch, ck.cfg.rec_chunk_tokens)
                rec = ck.recorder.attempt
            stats = ck.refresh_labels() if phase == "onpolicy" else None
            if getattr(trainer, "is_global_zero", True):
                warn = label_supply_warning(ck, stats) if phase == "onpolicy" else None
                self._append("epochs.jsonl", {"event": "epoch_start", "epoch": ck.epoch, "phase": phase,
                                              "recording": rec is not None, "attempt_rank0": rec,
                                              "labels": stats, "label_warning": warn, "mb": ck.mb,
                                              "gstep": int(trainer.global_step),
                                              "world_size": ck.world_size, "time": time.time(),
                                              "ema": ck.ema.state_dict(),
                                              "bev_ctrl": None if ck.bev_ctrl is None else ck.bev_ctrl.state_dict(),
                                              "cum": dict(ck.cum)})
                if warn and warn.get("warn"):
                    print("\n" + "!" * 100 + f"\n[ck_e2e2] WARNING epoch {ck.epoch}: {warn['msg']}\n" + "!" * 100,
                          flush=True)

        def on_train_epoch_end(self, trainer, pl_module):
            ck = agent._ck_e2e2
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
            ck = agent._ck_e2e2
            nb = trainer.num_training_batches
            nb = int(nb) if nb and math.isfinite(float(nb)) else 0
            ck.epoch_frac = float(trainer.current_epoch) + (float(batch_idx) / nb if nb > 0 else 0.0)
            ck.batch_idx = int(batch_idx)
            ck.gstep = int(trainer.global_step)
            ck.rank = int(getattr(trainer, "global_rank", 0))
            self._t_start = time.time()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            ck = agent._ck_e2e2
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
                    if k.startswith(STEP_LOG_PREFIXES):
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
            ck = agent._ck_e2e2
            allp = [p for p in pl_module.parameters() if p.grad is not None]
            if allp:
                gn_all = torch.stack([g.float() for g in torch._foreach_norm([p.grad for p in allp])])
                if not bool(torch.isfinite(gn_all).all()):
                    for p in allp:
                        p.grad = None
                    ck.skipped_steps += 1
                    if getattr(trainer, "is_global_zero", True):
                        print(f"[ck_e2e2] non-finite gradient at global_step {trainer.global_step}: optimiser step "
                              f"skipped (total {ck.skipped_steps})", flush=True)
                    return
            clip = float(ck.cfg.clip)
            params = [p for p in ck2_parameters(agent) if p.grad is not None]
            if clip > 0 and params:
                ck.last_clip_norm = float(torch.nn.utils.clip_grad_norm_(params, clip))

        # ---- end
        def _close(self):
            if self._f is not None:
                self._f.close()
                self._f = None

        def on_train_end(self, trainer, pl_module):
            ck = agent._ck_e2e2
            if ck.recorder is not None:
                ck.recorder.flush()
                ck.recorder = None
            self._close()
            if getattr(trainer, "is_global_zero", True):
                (self._io() / "TRAIN_DONE").write_text(json.dumps({"time": time.time(), "epoch": ck.epoch,
                                                                   "gstep": int(trainer.global_step), "mb": ck.mb}))

        def on_exception(self, trainer, pl_module, exception):
            ck = agent._ck_e2e2
            try:
                if ck.recorder is not None:
                    ck.recorder.flush()
                    ck.recorder = None
            finally:
                self._close()

    return CKE2E2Callback()
