"""refiner v4 network options (V4Cfg).  Stored as config.json 'v4' (teachers) / ck_e2e2.v4 (student); absent or {}
= V4Cfg() = off, i.e. the CK2 / refiner-v3 CKNet (mean readout, shared score head) bit for bit.

readout
  'mean'    CK2: h = [x_draft || mean_j x_station_j] -> shared score head (5 keys) + lat_head (unchanged).
  'hazard'  v4 I1 (report 72 / 77 Q3):
            * scene reading per PARENT path: corridor (48 stations x 17 lateral offsets) + 4 layers on the parent's
              draft (time_free: the parent's own arrival-time / speed geo channels are zeroed, so the station
              features describe the scene along the path, not a schedule);
            * every candidate c reads its parent's stations over the time it spends in each station's bin
              [s_j - ds / 2, s_j + ds / 2): entry time t_in,c(j), dwell w_c(j) = min(t_leave, horizon) - t_in, speed
              v_c(j) and lateral offset d_c(j) (sibling.on_parent: Frenet projection of c's poses on the parent's
              station frame).  A station is read when the candidate enters its bin within `horizon`, so station 0 is
              always read (no candidate reads nothing; a stopped candidate reads the bin it stops in with a long
              dwell);
            * per-key station hazard a_k(j) = MLP_k(x'_j, Hc(j, d_c(j)), phi(t_in,c(j)), w_c(j), v_c(j), e_c) for
              NC / TTC, and MLP_dac(x'_j, Hc(j, d_c(j)), d_c(j), e_c) for DAC (no time); e_c = the candidate's
              feasibility from the ego state (v0, planned first-segment speed change, stopping deceleration
              v0^2 / 2 S_8: plans the LQR tracker cannot follow overshoot them);
            * noisy-OR over the read stations: log P_pass,k = sum_j m_c(j) log sigmoid(-a_k(j))
              (0 learned parameters; the metric is 'fails if it fails anywhere');
            * EP head on [g_c, parent draft token, masked station mean, log1p S_8, log1p P_ref, S_8 / max(P_ref, S_8,
              5 m)] (P_ref detached), comfort head on the candidate's kinematics only (g_c = draft_mlp(u_d,c)),
            * lateral head on [g_c, hazard-weighted station features] (weights = softmax_j of the detached DAC
              hazard over the reached stations), zero-initialised last layer (identity decode at init).
            score_logit keeps the CK_KEYS layout [.., 5] (nc, dac, ep, ttc, comfort), so every loss / selection / eval
            path is unchanged.
  'evidence' v4 report 77 Q4-2 (evidence-decomposed critic): the same parent pass; the readout first predicts SCENE
            quantities on the parent's station grid -- occupancy logits O[j, i, m] (other objects at station j, lateral
            offset i, time bin m = 0.5 (m + 1) s) and the drivable margin D[j, i] (signed distance to the DAC boundary,
            m) -- then turns them into per-candidate evidence with official-rule-shaped approximations (evidence.py
            docstring lists the differences): NC = noisy-OR of O over the ego footprint (9 points) at the candidate's
            knots, TTC = the same over the footprint projected v * {0.3, 0.6,
            0.9} s ahead at t + delta, DAC = noisy-OR of sigmoid(-D / eps) over the 4 corners; + a zero-initialised
            residual MLP per key (what the rules do not explain); EP / comfort / lateral heads as 'hazard' (the
            lateral weights from the margin along the path).  Outputs O / D for the auxiliary losses (GT objects /
            the official drivable SDF on the parent grid points) and as the per-candidate 'reason'.
  'range'   v4_3 M4 allowed-range critic (docs/refiner_v4/V4_3_SPEC.md; ck/v4/range_readout.py docstring): the same
            parent pass, scene fields per UNIQUE parent on its station grid -- B blocked logits [48, 17, M] (the
            official ego box with its rear axle at the cell overlaps a not-behind object in time bin m of ev_tbin_s),
            U overlap logits [48, 17, M] (the same without the not-behind filter: lets TTC apply the not-behind rule
            from the CURRENT ego), D margin [48, 17] (m) -- read at each candidate's executed states (exec_model
            'lqr': the official LQR tracker port ck/v4/lqr.py, 41 states; 'plan': the 9 planned knots): NC / TTC /
            DAC = soft max over the states of B / B-or-U at the 0 / 0.3 / 0.6 / 0.9 s projections / -D / eps at the
            4 corners, + time / margin slacks and the critical station / time from the thresholded fields, + a
            zero-initialised residual; EP head optionally on the detached evidence (ep_evidence), lateral head on
            margin-weighted station features.  Needs ev_tbins * ev_tbin_s = 5.0 s (the GT range field's bins,
            range_targets.T_END): V4Cfg.range_arm() gives the spec setting (ev_tbins 20 of 0.25 s, exec 'lqr',
            ep_evidence, P_ref head).
pref_head   one learned query over the 200 scene tokens (attention pooling) + ego (v0, a0, command) -> log1p(P_ref)
            per token, P_ref = labels 'pdm_progress_eff' (token constant).  Used by the hazard EP head (detached); its
            own loss is lambda_pref * Huber (callers).
v4_3 options (all OFF / unused by default; only 'range' reads them, validate() rejects a non-default value of any of
them (V43_KEYS) with another readout).  readout 'range' alone keeps exec_model 'plan' and ep_evidence off (new options
default off); the spec arm (lqr, ep_evidence, 20 x 0.25 s, P_ref head) is V4Cfg.range_arm(), which callers building a
range critic should use:
  exec_model 'plan' | 'lqr'   candidate states the range readout reads (the edits always use 'lqr').
  ev_tbin_s                   range time bin width [s] (bin m = [m tbin, (m + 1) tbin), range_targets convention).
  ep_evidence                 EP head input += detached [lp_nc, lp_ttc, lp_dac, slack_nc, slack_ttc, slack_dac,
                              executed 4 s progress ratio, margin-weighted station features] (report 80 E1).
  vmax 'off' | 'feature'      + the executed curvature overspeed max_n relu(|v_n| - v_max(station)) as a residual input
                              (v_max from the predicted margins, range_targets.v_max).
  edits 'off' | 'computed'    CKNet.computed_edits (ck/v4/edits.py, spec M5): edit_k extra candidates from the
                              critic's top-edit_k parents, lon edit until the executed arrival at the critical station
                              is >= t_out_block + edit_dt, lat edit (either side, the smaller shift) until the executed
                              min corner margin >= edit_dlat (m).
  rg_read 'nearest' | 'any4'  how the range readout reads B / U at an executed point (Phase 0b R2 (2)): 'nearest' (the
                              v4_3 default) = the nearest cell; 'any4' = OR over the 2 x 2 grid points around the read
                              point (range_targets._cells 'any4'): the thresholded slacks / critical points OR the 4
                              cells' profiles (= range_targets.executed_slack(read='any4')), the differentiable
                              evidence soft-ORs them (the 4 distinct cells join the soft max over the states).  edits
                              'computed' reads the same way (ck/v4/edits.py: any4 tube check, critical reads and
                              their 4-cell windows).
  rg_stopped_aware            Phase 0b R2 (1): the GT field aux of this readout trains B on the stopped-aware B
                              (range_targets.blocked_field stopped_aware: stopped / static objects without the
                              not-behind filter; read by tools/ck/train_ck2.py (teachers) and online2 v4.rg_gt_aux
                              (student)), and the readout gets a THIRD field group Bp = the plain not-behind B (target
                              LON >= BEHIND_M of the same GT call) that every TTC read of B uses (delta 0 and
                              |v| delta < 2 m; NC reads the stopped-aware B, U unchanged): R2 changes NC only, the
                              official TTC rule has no stopped-track exception (range_readout docstring).  blk_head
                              then outputs 3 x ev_tbins (+ rg_hidden * ev_tbins + ev_tbins parameters, 1,100 at the
                              spec sizes; inside the CK2 budget).  edits 'computed' reads the three groups with the
                              same channel rule (ck/v4/edits.py: NC windows on B, TTC on Bp / U).
Head sizes (haz_hidden, ep_hidden, comfort_hidden, lat_hidden, pref_kdim, pref_hidden, ev_hidden, rg_hidden,
ep_ev_hidden): the defaults give the hazard + P_ref (the evidence + P_ref, the range + P_ref with or without
ep_evidence) critic the trainable parameter count of the CK2 critic it replaces (CK2 as trained: 3,391,755 = the
shared trunk 3,191,808 + score head 99,845 + lateral head 100,102; the new heads together take the 199,947 of the two
replaced heads, tests/test_v4_model.py::test_param_budget, tests/test_v4_range_readout.py::test_param_budget_range), so
a v4 vs CK2 comparison is not a capacity comparison (report 74, review 79 F7).  ep_ev_hidden (61) is the EP width with
ep_evidence: its 906 inputs then take the parameters of the 579-input, 96-wide head (55,389 vs 55,777).
Versions: 'v4_3' (this code) adds the options above; from_dict also reads 'v4_2' dicts (the trained v4T10_haz /
v4M10_haz teachers), whose missing v4_3 keys take the defaults = the v4_2 network bit for bit, as a 'v4_3' config.
v4_3 dicts written before rg_read / rg_stopped_aware existed load with their defaults (nearest, False) = unchanged.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

READOUTS = ("mean", "hazard", "evidence", "range")
CONFIG_VERSION = "v4_3"          # v4_3: range readout (M4) + computed edits (M5); v4_2: interval station reads +
                                 # feasibility, matched head sizes, 10 ev bins
READ_VERSIONS = ("v4_2", CONFIG_VERSION)        # from_dict accepts these (and upgrades to CONFIG_VERSION)
EXEC_MODELS = ("plan", "lqr")
VMAX_MODES = ("off", "feature")
EDIT_MODES = ("off", "computed")
RG_READS = ("nearest", "any4")
RANGE_T_END = 5.0                # [s] range time bins cover [0, 5.0 s) (= range_targets.T_END, the GT field's horizon)
MAX_TBINS = 20


@dataclass(frozen=True)
class V4Cfg:
    readout: str = "mean"
    pref_head: bool = False
    time_free: bool = True             # hazard: zero the parent's t_arr / v_arr geo channels
    lat_haz: bool = True               # hazard: hazard-localised lateral head (False: lateral head on [g_c, mean x'])
    horizon: float = 4.0               # [s] stations first reached later than this are not read (label horizon)
    n_freq: int = 4                    # time embedding: t/4, sin / cos(2 pi m t / 8), m = 1..n_freq
    haz_hidden: int = 48
    ep_hidden: int = 96
    comfort_hidden: int = 32
    lat_hidden: int = 128
    pref_kdim: int = 64
    pref_hidden: int = 52
    ev_hidden: int = 56                # evidence: hidden width of the occupancy / margin / residual MLPs
    prior_n_eff: float = 24.0          # set_score_prior: per-station hazard prior = 1 - p^(1 / n_eff)
    ev_tbins: int = 10                 # evidence: occupancy time bins t = 0.5 (m + 1) s, m = 0..ev_tbins-1 (<= 10: to
                                       # 5.0 s, the official TTC reads object states up to 4.9 s); range: M bins of
                                       # ev_tbin_s with M ev_tbin_s = 5.0 s (<= 20)
    ev_eps: float = 0.3                # evidence / range: initial margin temperature [m] of p_out = sigmoid(-D / eps)
    ev_resid: bool = True              # evidence / range: zero-initialised residual MLP per key
    corrector: bool = False            # learned corrector A on the critic's top-corr_k (needs a parent-pass readout)
    corr_k: int = 8
    # ---- v4_3 (readout 'range' / computed edits; docstring 'v4_3 options')
    exec_model: str = "plan"           # range: candidate states read ('plan' 9 knots | 'lqr' 41 tracked states)
    ev_tbin_s: float = 0.25            # range: time bin width [s]
    ep_evidence: bool = False          # range: EP head on the detached evidence + executed progress ratio
    vmax: str = "off"                  # range: 'feature' = executed curvature overspeed as a residual input
    edits: str = "off"                 # 'computed' = CKNet.computed_edits (needs readout 'range')
    edit_k: int = 8                    # computed edits: shortlist size (the corrector's K slots)
    edit_dt: float = 0.2               # [s] lon edit: executed arrival >= t_out_block + edit_dt at the critical cell
    edit_dlat: float = 0.3             # [m] lat edit: executed min corner margin >= edit_dlat
    rg_hidden: int = 54                # range: hidden width of the B / U, D and residual MLPs
    ep_ev_hidden: int = 61             # range: EP head width when ep_evidence (budget-matched, docstring)
    rg_read: str = "nearest"           # range: B / U read at the executed points ('nearest' | 'any4', Phase 0b R2)
    rg_stopped_aware: bool = False     # range: B target = the stopped-aware B (R2) + a Bp group for TTC (docstring)
    version: str = CONFIG_VERSION

    @property
    def on(self) -> bool:
        return self.readout != "mean" or self.pref_head or self.corrector

    @property
    def hazard(self) -> bool:
        return self.readout == "hazard"

    @property
    def evidence(self) -> bool:
        return self.readout == "evidence"

    @property
    def range(self) -> bool:
        return self.readout == "range"

    @property
    def computed_edits(self) -> bool:
        return self.edits == "computed"

    @property
    def parent_pass(self) -> bool:
        """readouts that read the scene on each candidate's parent corridor (CKNet.score_set's v4 path)."""
        return self.readout in ("hazard", "evidence", "range")

    @classmethod
    def range_arm(cls, **kw) -> "V4Cfg":
        """The spec M4 / M5 setting of the range critic (V4_3_SPEC.md): readout 'range', exec_model 'lqr', 20 bins of
        0.25 s, ep_evidence, P_ref head (feeds both EP ratios); kw override any field (e.g. edits='computed')."""
        base = dict(readout="range", exec_model="lqr", ev_tbins=20, ev_tbin_s=0.25, ep_evidence=True, pref_head=True)
        base.update(kw)
        return cls(**base).validate()

    def validate(self, temporal_on: bool = False) -> "V4Cfg":
        if self.readout not in READOUTS:
            raise ValueError(f"v4.readout {self.readout!r} not in {READOUTS}")
        if self.parent_pass and temporal_on:
            raise ValueError(f"v4.readout {self.readout!r} is not combined with the refiner-v3 temporal branch")
        if self.corrector and not self.parent_pass:
            raise ValueError("v4.corrector needs a parent-pass readout ('hazard' / 'evidence' / 'range'; its context "
                             "is the parent pass)")
        if not (1 <= int(self.corr_k) <= 32):
            raise ValueError(f"v4.corr_k {self.corr_k} must be in [1, 32]")
        if not (1 <= int(self.ev_tbins) <= MAX_TBINS) or not (0.01 <= float(self.ev_eps) <= 5.0):
            raise ValueError(f"v4: ev_tbins {self.ev_tbins} must be in [1, {MAX_TBINS}], ev_eps {self.ev_eps} in "
                             f"[0.01, 5]")
        if self.readout != "range" and int(self.ev_tbins) > 10:
            raise ValueError(f"v4: ev_tbins {self.ev_tbins} must be in [1, 10] for readout {self.readout!r} (0.5 s "
                             f"bins: 10 reach 5.0 s)")
        if not (0.05 <= float(self.ev_tbin_s) <= RANGE_T_END):
            raise ValueError(f"v4.ev_tbin_s {self.ev_tbin_s} outside [0.05, {RANGE_T_END}] s")
        if self.readout == "range" and abs(int(self.ev_tbins) * float(self.ev_tbin_s) - RANGE_T_END) > 1e-6:
            raise ValueError(f"v4 readout 'range' needs ev_tbins x ev_tbin_s = {RANGE_T_END} s (the GT range field's "
                             f"bins, range_targets.T_END), got {self.ev_tbins} x {self.ev_tbin_s}: e.g. ev_tbins 20 at "
                             f"0.25 s, or V4Cfg.range_arm()")
        if not (0.5 <= float(self.horizon) <= 8.0):
            raise ValueError(f"v4.horizon {self.horizon} outside [0.5, 8] s")
        for name, val, allowed in (("exec_model", self.exec_model, EXEC_MODELS), ("vmax", self.vmax, VMAX_MODES),
                                   ("edits", self.edits, EDIT_MODES), ("rg_read", self.rg_read, RG_READS)):
            if val not in allowed:
                raise ValueError(f"v4.{name} {val!r} not in {allowed}")
        # edits 'computed' reads the fields with rg_read / rg_stopped_aware as the readout does (ck/v4/edits.py header)
        if self.readout != "range":     # every v4_3 field is range-only: a non-default value elsewhere is an error
            dflt = {f.name: f.default for f in dataclasses.fields(V4Cfg)}
            on = [k for k in V43_KEYS if getattr(self, k) != dflt[k]]
            if on:
                raise ValueError(f"v4: {on} only apply to readout 'range' (got {self.readout!r}; non-default "
                                 + ", ".join(f"{k}={getattr(self, k)!r}" for k in on) + ")")
        if not (1 <= int(self.edit_k) <= 32):
            raise ValueError(f"v4.edit_k {self.edit_k} must be in [1, 32]")
        if not (0.0 <= float(self.edit_dt) <= 2.0) or not (0.0 <= float(self.edit_dlat) <= 2.0):
            raise ValueError(f"v4: edit_dt {self.edit_dt} s / edit_dlat {self.edit_dlat} m outside [0, 2]")
        sizes = ("haz_hidden", "ep_hidden", "comfort_hidden", "lat_hidden", "pref_kdim", "pref_hidden", "ev_hidden",
                 "rg_hidden", "ep_ev_hidden")
        if int(self.n_freq) < 0 or any(int(getattr(self, k)) < 8 for k in sizes):
            raise ValueError(f"v4: bad sizes n_freq {self.n_freq} "
                             + " ".join(f"{k} {getattr(self, k)}" for k in sizes))
        if float(self.prior_n_eff) < 1.0:
            raise ValueError(f"v4.prior_n_eff {self.prior_n_eff} < 1")
        if self.version != CONFIG_VERSION:
            raise ValueError(f"v4.version {self.version!r} != {CONFIG_VERSION!r} (from_dict reads {READ_VERSIONS})")
        return self

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "V4Cfg":
        """None / {} -> off.  Unknown keys -> ValueError.  Bools given as strings ('true' / '0') are parsed.
        version 'v4_2' (or absent) is read as the current version: its dicts have none of the v4_3 keys, which take
        their defaults (= the v4_2 network); a 'v4_2' dict that carries a v4_3 key is an error."""
        if not d:
            return cls()
        d = dict(d)
        names = {f.name for f in dataclasses.fields(cls)}
        bad = sorted(set(d) - names)
        if bad:
            raise ValueError(f"v4: unknown key(s) {bad}")
        ver = d.get("version")
        if ver is not None:
            if str(ver) not in READ_VERSIONS:
                raise ValueError(f"v4.version {ver!r} not in {READ_VERSIONS}")
            if str(ver) == "v4_2":
                late = sorted(set(d) & set(V43_KEYS))
                if late:
                    raise ValueError(f"v4: a 'v4_2' config with v4_3 key(s) {late}")
            d["version"] = CONFIG_VERSION
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


# keys a 'v4_2' config.json never carries (from_dict); the size fields are listed too (v4_2 had no range heads)
V43_KEYS = ("exec_model", "ev_tbin_s", "ep_evidence", "vmax", "edits", "edit_k", "edit_dt", "edit_dlat", "rg_hidden",
            "ep_ev_hidden", "rg_read", "rg_stopped_aware")
