"""refiner v5 options (V5Cfg; design: SSR/report/97_refiner_v5_design.md, user's add.md v5 structure).  Stored as
config.json 'v5' (teachers); absent / {} = V5Cfg() = off = the v4_3 network bit for bit (no module, no parameter, no
RNG draw, no output key).  v5 needs the v4_3 range readout (V4Cfg.readout 'range' with its residual MLP).

relation    'off' | 'teacher'   object channel (ck/v5/relation.py): per-object relation r_i of the t0 object rows with
            every candidate's executed states (exec_model 'lqr' | 'ref'; SAT overlap, modes hold / constant velocity / constant acceleration
            from the causal past frames), learned reliability a_i, per-state attributed hazard H_j (NC, TTC).  Adds 4
            residual inputs (lp_rel_nc, lp_rel_ttc, max_i a_i r_i, sum_i a_i) and a direct log-pass term per key
            (zero-initialised weights).  'teacher' = BEVFusion detections + past frames (refiner-v3 object pack,
            OBJ_INPUT_KEYS); 'student' = the e2e student's own det head (ck/v5/student_objects.py: boxes, velocities,
            det_hidden; detached; no CA mode).
unknown_ev  an executed state whose NC point or any corner is off the parent grid has NO local evidence ("unknown", never
            "safe"): the fraction of such states joins the residual input and the per-state reads Z.
corrector   corrector A (ck/v5/corrector.py): 12 decoder controls (z_lon 6 + w_lat 6, decoder mode B) for given draft
            columns from the per-state reads Z (NOT detached: the correction loss trains the reader, add.md (3)) + the
            critic context (detached, as v4 A) + the top-3 related objects; tau1 = decode(tau0, mu_lon, w_lat_head +
            mu_lat) (the lateral head is folded in: at init tau1 = today's laterally corrected draft).  Trained by the
            executed-state GT surrogate L_refine(X1), X1 = exec(tau1) (ck/v5/refine_loss.py).
z_channels  which per-state reads the corrector sees: 'all'; 'det' (b_nc, b_ttc, unknown, H -- a DET teacher's D field
            is unsupervised); 'map' (d_min, unknown -- a MAP teacher's B / U fields are unsupervised); the other channels
            are zeroed (no gradient into fields nothing else trains; review 2026-10-20 f11 / f24).
z_stride    per-state reads Z at every z_stride-th executed state (2: t = 0, 0.2, .., 4.0 s, 21 states); the relation runs on
            the same states.
rel_pstop   relation 'student' only: the reliability MLP a_i also reads p_stop_i = sigmoid(stop logit) of the student's
            det-head V1c stop_branch (design §3.3 a_i = sigma(MLP_a([det_hidden, score, class, range, p_stop])); the
            ACC head's p_stop2s is an aux target only, report 97 §1 (c)/(d)): one more feats column (N_FEAT_S + 1).
            Needs agent.config.det_stop_branch (online2 refuses it otherwise).
rel_modes   relation 'student' only (design §3.2 modes): 'cv' = hold + constant velocity (+ CA from the past rows with
            rel_ca, teacher); 'cv+fut' = + the det head's own futures m1..6 (traj_preds, pi = softmax(traj_cls) mixed with
            a learned CV prior pi0 = sigmoid(cv_prior), init 0.5; ck/v5/relation.py); 'cv+fut+ca' (student m7 constant
            acceleration) is NOT implemented -- conditional on PT-11 / PT-B3 (design §3.2 / §3.3), refused.
top_feats   corrector top-3 object features (relation + corrector): 'v1' = [a r_nc, a r_ttc, x, y, vx, vy, t*] (7, the
            teachers' A_T); 'v2' (design §3.7 A_S input) = + station s*, side, key (relation.py) + Delta r of the probes
            a-0.5 / l-0.5 / l+0.5 of the draft (CKNet._v5_act; 13 per object).  The probes are the CK2 variant rule
            (ck/variants.make_variants) with probe_s_on_frac / probe_dtype = the run's variants config (online2.v5_net).
z_adapt     the student's Z adapters P_det / P_map (ck/v5/zadapt.py; design §2.1 L82, §3.8 L191): the corrector reads
            the per-state reads in the TEACHERS' normalised units (frozen per-channel mu_T / sigma_T of the teacher
            readers, buffers set from tools/ck/v5/fit_z_stats.py records) through a per-channel affine + one
            z_adapt_hidden-wide MLP per half (zero-initialised: identity on the normalised reads at init); Z-KD runs in
            the same units (online2._v5_terms).  Needs the corrector.  Off (default) = no module, no buffer.  The
            teachers' A_T is trained without it (a teacher adapter flag is NOT implemented, main-session decision D9).
Sizes: n_obj (t0 rows read, score order), rel_hidden (reliability MLP), corr_hidden / z_hidden (corrector).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

RELATIONS = ("off", "teacher", "student")
CONFIG_VERSION = "v5_1"
N_Z_DENSE = 4            # per-state reads of the range fields: b_nc, b_ttc, d_min, unknown (ck/v5/zread.py)
N_Z_REL = 2              # + relation hazard H_nc, H_ttc
N_REL_EV = 4             # relation residual inputs
N_TOP = 3                # top related objects fed to the corrector
N_TOP_FEAT = 7           # a r_nc, a r_ttc, x / 32, y / 32, vx / 10, vy / 10, t* / 4  (top_feats 'v1')
N_TOP_FEAT_V2 = 13       # 'v2': + station / 32, side, key (relation) + Delta r of the 3 probes (corrector input)
REL_MODES = ("cv", "cv+fut")
TOP_FEATS = ("v1", "v2")
# keys added after the running teachers were written: left out of to_dict() at their default, so a teacher's
# config.json 'v5' stays as before (from_dict of an old dict gives the default anyway)
_OMIT_AT_DEFAULT = ("rel_modes", "top_feats", "probe_s_on_frac", "probe_dtype", "z_adapt", "z_adapt_hidden")


@dataclass(frozen=True)
class V5Cfg:
    relation: str = "off"
    n_obj: int = 96
    rel_ca: bool = True
    rel_eps: float = 0.3
    rel_hidden: int = 16
    unknown_ev: bool = False
    corrector: bool = False
    corr_hidden: int = 256
    z_hidden: int = 32
    z_stride: int = 2
    z_channels: str = "all"        # corrector input channels: 'all' | 'det' (no D) | 'map' (D + unknown only)
    rel_pstop: bool = False        # relation 'student': p_stop_i of the det head's stop_branch in the a_i features
    rel_modes: str = "cv"          # relation 'student': 'cv' | 'cv+fut' (the det head's futures as modes m1..6)
    top_feats: str = "v1"          # corrector top-3 features: 'v1' (7) | 'v2' (13: + station, side, key, Delta r)
    probe_s_on_frac: float = 0.2   # top_feats 'v2' Delta r probes: the run's variants s_on_frac / compute dtype
    probe_dtype: str = "float64"
    z_adapt: bool = False          # student Z adapters P_det / P_map on teacher-normalised reads (ck/v5/zadapt.py)
    z_adapt_hidden: int = 32
    version: str = CONFIG_VERSION

    @property
    def on(self) -> bool:
        return self.relation != "off" or self.unknown_ev or self.corrector

    @property
    def rel(self) -> bool:
        return self.relation != "off"

    @property
    def n_extra(self) -> int:
        """residual inputs added to the range readout."""
        return (N_REL_EV if self.rel else 0) + (1 if self.unknown_ev else 0)

    @property
    def n_z(self) -> int:
        return N_Z_DENSE + (N_Z_REL if self.rel else 0)

    @property
    def n_top_feat(self) -> int:
        """corrector input features per top object (top_feats)."""
        return N_TOP_FEAT_V2 if self.top_feats == "v2" else N_TOP_FEAT

    def validate(self, v4=None) -> "V5Cfg":
        if self.relation not in RELATIONS:
            raise ValueError(f"v5.relation {self.relation!r} not in {RELATIONS}")
        if not (1 <= int(self.n_obj) <= 96):
            raise ValueError(f"v5.n_obj {self.n_obj} not in [1, 96]")
        if not (0.01 <= float(self.rel_eps) <= 5.0):
            raise ValueError(f"v5.rel_eps {self.rel_eps} not in [0.01, 5] m")
        if self.rel_pstop and self.relation != "student":
            raise ValueError(f"v5.rel_pstop reads the student det head's stop_branch: relation 'student' only, got "
                             f"{self.relation!r}")
        if self.rel_modes == "cv+fut+ca":
            raise ValueError("v5.rel_modes 'cv+fut+ca' (the student's m7 constant-acceleration mode) is conditional on "
                             "PT-11 / PT-B3 (design §3.2 / §3.3): not implemented")
        if self.rel_modes not in REL_MODES:
            raise ValueError(f"v5.rel_modes {self.rel_modes!r} not in {REL_MODES}")
        if self.rel_modes != "cv" and self.relation != "student":
            raise ValueError(f"v5.rel_modes {self.rel_modes!r} reads the student det head's futures: relation "
                             f"'student' only, got {self.relation!r}")
        if self.top_feats not in TOP_FEATS:
            raise ValueError(f"v5.top_feats {self.top_feats!r} not in {TOP_FEATS}")
        if self.top_feats != "v1" and not (self.rel and self.corrector):
            raise ValueError("v5.top_feats 'v2' is a corrector input of the relation channel: needs relation and "
                             "corrector")
        if not (0.0 < float(self.probe_s_on_frac) < 1.0) or self.probe_dtype not in ("float64", "float32"):
            raise ValueError(f"v5.probe_s_on_frac {self.probe_s_on_frac} / probe_dtype {self.probe_dtype!r}")
        if self.z_adapt and not self.corrector:
            raise ValueError("v5.z_adapt adapts the corrector's input Z: needs v5.corrector (an unused adapter breaks "
                             "DDP)")
        if int(self.z_adapt_hidden) < 4:
            raise ValueError(f"v5.z_adapt_hidden {self.z_adapt_hidden} must be >= 4")
        if self.z_channels not in ("all", "det", "map"):
            raise ValueError(f"v5.z_channels {self.z_channels!r} not in ('all', 'det', 'map')")
        if int(self.z_stride) not in (1, 2, 4, 5):
            raise ValueError(f"v5.z_stride {self.z_stride} not in (1, 2, 4, 5) (41 LQR states)")
        if min(int(self.rel_hidden), int(self.corr_hidden), int(self.z_hidden)) < 4:
            raise ValueError("v5: hidden sizes must be >= 4")
        if self.version != CONFIG_VERSION:
            raise ValueError(f"v5.version {self.version!r} != {CONFIG_VERSION!r}")
        if self.on and v4 is not None:
            if not getattr(v4, "range", False):
                raise ValueError("v5 needs the v4_3 range readout (V4Cfg.readout 'range')")
            if not bool(getattr(v4, "ev_resid", True)):
                raise ValueError("v5 needs the range readout's residual MLP (V4Cfg.ev_resid)")
            if getattr(v4, "exec_model", "lqr") not in ("lqr", "ref"):
                raise ValueError("v5 reads 41 states (V4Cfg.exec_model 'lqr' = the tracked rollout | 'ref' = its "
                                 "reference, the no-LQR student)")
            if getattr(v4, "corrector", False):
                raise ValueError("v5.corrector replaces the v4 corrector (V4Cfg.corrector must be off)")
        return self

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        for k in _OMIT_AT_DEFAULT:
            if d[k] == getattr(V5Cfg, k):
                del d[k]
        return d

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "V5Cfg":
        """None / {} -> off.  Unknown keys -> ValueError; bools given as strings are parsed."""
        if not d:
            return cls()
        d = dict(d)
        names = {f.name for f in dataclasses.fields(cls)}
        bad = sorted(set(d) - names)
        if bad:
            raise ValueError(f"v5: unknown key(s) {bad}")
        kw = {}
        for f in dataclasses.fields(cls):
            if f.name not in d or d[f.name] is None:
                continue
            v = d[f.name]
            if f.type in ("bool", bool):
                v = v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")
            elif f.type in ("int", int):
                v = int(v)
            elif f.type in ("float", float):
                v = float(v)
            else:
                v = str(v)
            kw[f.name] = v
        return cls(**kw).validate()
