"""refiner-v3 temporal branch: constants shared by every builder and the run configuration TemporalCfg.

Pinned by docs/refiner_v3/SPEC.md section 4.1 (values copied verbatim; changing one is a spec change).

Object rows (section 3.2): D_OBJ = 17 float32, metric, t0 N frame (rear-axle origin, x fwd, y left), unnormalised:
  0 x  1 y  2 cos h  3 sin h  4 L  5 W  6 vx  7 vy  8 score  9..15 class one-hot (BEVFusion labels 0..6)  16 age_s
Frames f = 0 (t0), 1 (t-0.5 s), 2 (t-1.0 s); K_OBJ = 96 slots per frame, sorted by score (padding = zero rows).
K_OBJ was raised 64 -> 96 in the review fix (docs/refiner_v3/REVIEW_FIXES.md D1): at 64, 2.62 % of the navtrain_train
frames were truncated (SPEC 3.4 limit 1 %), at 96 0.58 % [실측 BUILD_data.md section 6].  The model is K-agnostic
(trim to the batch maximum); K only sets the pack layout, so the pack root carries it (PACK_ROOT = D3/objpack_k96).
Groups: GROUP_OF_CLASS maps a class to 0 vehicle / 1 VRU / 2 static; "agent" = classes 0-2 (groups 0, 1).
Model-side normalisation uses the fixed NORM constants only (never statistics).

TemporalCfg (frozen dataclass)
  on          False = the CK2 code path, bit-identical to 8ae8b1d (no module, parameter, buffer, key or RNG draw)
  source      'bevfusion' (DET teacher, Step 1) | 'v2det' (student, later) | 'gt' (oracle ablation, later)
  forecaster  'learned' (arm a) | 'ca' (arm b, analytic constant acceleration) | 'oracle_gt' (arm d, PRIVILEGED)
  n_modes     learned: 1..8 (default 6); ca / oracle_gt: 1
  lambda_fc   weight of forecast_loss in the total loss (learned only)
  fc_backprop False (default): raster / gap scalars read fc.detach(), the critic loss never reaches the forecaster
  p_frame_drop, p_obj_drop, v_noise : training augmentation (tools/ck/data/obj_pack.augment_objects)
  pack_root, ca_scales : data paths; version : config schema version
  fc_init     learned arm: path of a pre-trained forecaster checkpoint (tools/ck/v3/pretrain_forecaster.py); loaded
              into temporal.forecaster by train_ck2 at build time ('' = random init, the old from-scratch behaviour)
  fc_train    learned arm: 'frozen' (default, review fix D3: the forecaster is fixed during critic training, no forecast
              loss, like the fixed CA forecaster of arm b) | 'joint' (optional: forecast loss lambda_fc keeps training
              it; with fc_backprop False the critic loss still never reaches it).  ca / oracle_gt: ignored ('frozen').
  privileged  property: on and (forecaster == 'oracle_gt' or source == 'gt')
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ----------------------------------------------------------------------------------------------- object rows
K_OBJ = 96
N_FRAMES = 3
D_OBJ = 17
N_CLS = 7
COL = dict(x=0, y=1, cos=2, sin=3, L=4, W=5, vx=6, vy=7, score=8, cls0=9, age=16)
SCORE_THR = (0.3, 0.2, 0.2)
DT_RANGE = (0.25, 1.25)
GROUP_OF_CLASS = (0, 1, 1, 2, 2, 2, 2)
AGENT_CLASSES = (0, 1, 2)                                   # 0 vehicle, 1 VRU, 2 static (groups)
N_GROUPS = 3

# ----------------------------------------------------------------------------------------------- forecast / raster
FC_STEPS = 10
FC_T = tuple(0.5 * k for k in range(1, 11))                 # forecast knots (s)
FC_DT = 0.5
N_RT = 21
R_T = tuple(0.25 * k for k in range(21))                    # raster time bins 0..5 s
EGO_L, EGO_W, EGO_OFF = 5.176, 2.297, 1.461                 # Pacifica, centre ahead of rear axle
EGO_REAR, EGO_FRONT = EGO_OFF - EGO_L / 2, EGO_OFF + EGO_L / 2   # -1.127, +4.049 (rear-axle s offsets)
TAPS = (0.0, 0.3, 0.6, 0.9)
TAP_T_MAX = 4.0                                             # pdm_scorer.py 470-506
TJ = tuple(0.25 * k for k in range(1, 17))                  # gap-scalar ego times (a3)
ROI = {"bevfusion": (0.0, -32.0, 32.0, 32.0)}               # x0, y0, x1, y1 (manifest point_cloud_range)
KNOWN_MARGIN = 0.5
LAT_DROP = 10.0
B0 = 0.1
B_MIN, B_MAX = 0.05, 20.0
B_ORACLE = 0.25                                             # Laplace scales (m)
NORM = dict(xy=32.0, v=15.0, L=10.0, W=5.0)

# ----------------------------------------------------------------------------------------------- seeds / paths / keys
TEMPORAL_SEED_OFFSET = 52361
FORECASTER_SEED_OFFSET = 60493
D3 = Path("/home/external-user/ssd/yongjae_refiner/ck/v3")
PACK_ROOT = D3 / f"objpack_k{K_OBJ}"                      # K in the name: a pack serves one layout
CA_SCALES = D3 / "ca_scales.json"
SOURCES = ("bevfusion", "v2det", "gt")
FORECASTERS = ("learned", "ca", "oracle_gt")
OBJ_INPUT_KEYS = ("obj_in", "obj_mask", "frame_valid")     # causal model inputs
PRIV_INPUT_KEYS = ("priv_fc_xyh", "priv_fc_state")         # oracle arm only (PRIVILEGED)
TARGET_KEYS = ("fc_xyh", "fc_state")                       # loss / metrics only, never forward()
FC_NONE, FC_OBS, FC_ABSENT, FC_BEYOND = 0, 1, 2, 3
FC_SINGLE = 4      # matched singleton GT track (one keyframe, at t0): the official rule holds it at that pose, so the
                   # oracle reads it as OBS; never a motion target (loss, metrics and CA scales use FC_OBS only)
FC_TRAIN = ("frozen", "joint")
CONFIG_VERSION = "v3_temporal_1"


@dataclass(frozen=True)
class TemporalCfg:
    on: bool = False
    source: str = "bevfusion"
    forecaster: str = "learned"
    n_modes: int = 6
    lambda_fc: float = 1.0
    fc_backprop: bool = False          # default: critic loss does NOT reach the forecaster (section 4.6)
    p_frame_drop: float = 0.1          # per past frame
    p_obj_drop: float = 0.05           # per object slot, every frame
    v_noise: float = 0.3               # m/s, N(0, v_noise^2) per velocity component
    pack_root: str = str(PACK_ROOT)
    ca_scales: str = str(CA_SCALES)
    version: str = CONFIG_VERSION
    fc_init: str = ""                  # learned: pre-trained forecaster checkpoint ('' = random init)
    fc_train: str = "frozen"           # learned: 'frozen' (default) | 'joint'

    # ------------------------------------------------------------------------------------------ derived
    @property
    def privileged(self) -> bool:
        return bool(self.on and (self.forecaster == "oracle_gt" or self.source == "gt"))

    @property
    def roi(self):
        """Source ROI (x0, y0, x1, y1) used by the raster's known channel."""
        if self.source not in ROI:
            raise NotImplementedError(f"no ROI defined for source {self.source!r}")
        return ROI[self.source]

    # ------------------------------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["privileged"] = self.privileged
        return d

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "TemporalCfg":
        """None / {} / missing 'on' -> off (old config.json files).  Unknown keys -> ValueError.  'privileged' is
        derived: if present it must agree with the derived value."""
        if not d:
            return cls()
        d = dict(d)
        priv = d.pop("privileged", None)
        if d.get("on") and d.get("forecaster", "learned") == "learned" and "fc_train" not in d:
            d["fc_train"] = "joint"      # configs written before fc_train existed trained the forecaster jointly
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(d) - names)
        if unknown:
            raise ValueError(f"TemporalCfg.from_dict: unknown keys {unknown}")
        types = {f.name: f.type for f in dataclasses.fields(cls)}
        kw = {}
        for k, v in d.items():
            t = types[k]
            if t in ("bool", bool):
                kw[k] = bool(int(v)) if isinstance(v, (int, float, str)) and not isinstance(v, bool) else bool(v)
            elif t in ("int", int):
                kw[k] = int(v)
            elif t in ("float", float):
                kw[k] = float(v)
            else:
                kw[k] = str(v)
        cfg = cls(**kw)
        if priv is not None and bool(priv) != cfg.privileged:
            raise ValueError(f"TemporalCfg.from_dict: stored privileged={priv} but derived {cfg.privileged}")
        return cfg

    def replace(self, **kw) -> "TemporalCfg":
        return dataclasses.replace(self, **kw)

    # ------------------------------------------------------------------------------------------ checks
    def validate(self, arm: str) -> None:
        """Raise unless this configuration can be built for CK arm `arm` in Step 1 (no-op when off)."""
        if not self.on:
            return
        if arm != "T":
            raise ValueError(f"temporal branch requires arm 'T' in Step 1 (MAP teacher: off; S arm later), got {arm!r}")
        if self.source not in SOURCES:
            raise ValueError(f"source {self.source!r} not in {SOURCES}")
        if self.source in ("v2det", "gt"):
            raise NotImplementedError(f"source {self.source!r} is not built in Step 1")
        if self.forecaster not in FORECASTERS:
            raise ValueError(f"forecaster {self.forecaster!r} not in {FORECASTERS}")
        if self.forecaster == "learned":
            if not 1 <= int(self.n_modes) <= 8:
                raise ValueError(f"learned forecaster needs 1 <= n_modes <= 8, got {self.n_modes}")
        elif int(self.n_modes) != 1:
            raise ValueError(f"forecaster {self.forecaster!r} needs n_modes == 1, got {self.n_modes}")
        for k in ("p_frame_drop", "p_obj_drop"):
            v = float(getattr(self, k))
            if not 0.0 <= v < 1.0:
                raise ValueError(f"{k} must be in [0, 1), got {v}")
        if float(self.v_noise) < 0 or float(self.lambda_fc) < 0:
            raise ValueError("v_noise and lambda_fc must be >= 0")
        if self.version != CONFIG_VERSION:
            raise ValueError(f"TemporalCfg version {self.version!r} != {CONFIG_VERSION!r}")
        if self.fc_train not in FC_TRAIN:
            raise ValueError(f"fc_train {self.fc_train!r} not in {FC_TRAIN}")
        if self.forecaster != "learned" and (self.fc_init or self.fc_train != "frozen"):
            raise ValueError(f"fc_init / fc_train apply to the learned forecaster only (got {self.fc_init!r}, "
                             f"{self.fc_train!r} for {self.forecaster!r})")
        if self.fc_backprop and self.fc_train == "frozen":
            raise ValueError("fc_backprop needs fc_train 'joint' (a frozen forecaster gets no gradient)")

    @property
    def fc_frozen(self) -> bool:
        """learned forecaster fixed during critic training (no forecast loss, no gradient, run under no_grad)."""
        return bool(self.on and self.forecaster == "learned" and self.fc_train == "frozen")

    def check_trainable(self) -> None:
        """Training-time rule (train_ck2): a frozen learned forecaster needs a pre-trained checkpoint (fc_init)."""
        if self.fc_frozen and not self.fc_init:
            raise ValueError("learned forecaster with fc_train 'frozen' needs fc_init (a pre-trained forecaster "
                             "checkpoint, tools/ck/v3/pretrain_forecaster.py); pass fc_train 'joint' to train it "
                             "with the critic instead")

    def normalised(self) -> "TemporalCfg":
        """OFF configs compare equal whatever their other fields (review fix: an OFF run with another default stays
        resumable / usable as init_from)."""
        return self if self.on else type(self)()
