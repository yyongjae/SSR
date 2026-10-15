"""refiner v4 network options (V4Cfg).  Stored as config.json 'v4' (teachers) / ck_e2e2.v4 (student); absent or {}
= V4Cfg() = off, i.e. the CK2 / refiner-v3 CKNet (mean readout, shared score head) bit for bit.

readout
  'mean'    CK2: h = [x_draft || mean_j x_station_j] -> shared score head (5 keys) + lat_head (unchanged).
  'hazard'  v4 I1 (report 72 / 77 Q3):
            * scene reading per PARENT path: corridor (48 stations x 17 lateral offsets) + 4 layers on the parent's
              draft (time_free: the parent's own arrival-time / speed geo channels are zeroed, so the station
              features describe the scene along the path, not a schedule);
            * every candidate c reads its parent's stations at its OWN arrival time t_c(j), speed v_c(j) and lateral
              offset d_c(j) (sibling.on_parent: Frenet projection of c's poses on the parent's station frame);
            * per-key station hazard a_k(j) = MLP_k(x'_j, Hc(j, d_c(j)), phi(t_c(j)), v_c(j)) for NC / TTC, and
              MLP_dac(x'_j, Hc(j, d_c(j)), d_c(j)) for DAC (no time);
            * noisy-OR over the stations reached within `horizon`: log P_pass,k = sum_j m_c(j) log sigmoid(-a_k(j))
              (0 learned parameters; the metric is 'fails if it fails anywhere');
            * EP head on [g_c, parent draft token, masked station mean, log1p S_8, log1p P_ref, S_8 / max(P_ref, S_8,
              5 m)] (P_ref detached), comfort head on the candidate's kinematics only (g_c = draft_mlp(u_d,c)),
            * lateral head on [g_c, hazard-weighted station features] (weights = softmax_j of the detached DAC
              hazard over the reached stations), zero-initialised last layer (identity decode at init).
            score_logit keeps the CK_KEYS layout [.., 5] (nc, dac, ep, ttc, comfort), so every loss / selection / eval
            path is unchanged.
pref_head   one learned query over the 200 scene tokens + ego (v0, a0, command) -> log1p(P_ref) per token, P_ref =
            labels 'pdm_progress_eff' (token constant).  Used by the hazard EP head (detached); its own loss is
            lambda_pref * Huber (callers).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

READOUTS = ("mean", "hazard")
CONFIG_VERSION = "v4_1"


@dataclass(frozen=True)
class V4Cfg:
    readout: str = "mean"
    pref_head: bool = False
    time_free: bool = True             # hazard: zero the parent's t_arr / v_arr geo channels
    lat_haz: bool = True               # hazard: hazard-localised lateral head (False: lateral head on [g_c, mean x'])
    horizon: float = 4.0               # [s] stations first reached later than this are not read (label horizon)
    n_freq: int = 4                    # time embedding: t/4, sin / cos(2 pi m t / 8), m = 1..n_freq
    haz_hidden: int = 128
    ep_hidden: int = 256
    prior_n_eff: float = 24.0          # set_score_prior: per-station hazard prior = 1 - p^(1 / n_eff)
    version: str = CONFIG_VERSION

    @property
    def on(self) -> bool:
        return self.readout != "mean" or self.pref_head

    @property
    def hazard(self) -> bool:
        return self.readout == "hazard"

    def validate(self, temporal_on: bool = False) -> "V4Cfg":
        if self.readout not in READOUTS:
            raise ValueError(f"v4.readout {self.readout!r} not in {READOUTS}")
        if self.hazard and temporal_on:
            raise ValueError("v4.readout 'hazard' is not combined with the refiner-v3 temporal branch")
        if not (0.5 <= float(self.horizon) <= 8.0):
            raise ValueError(f"v4.horizon {self.horizon} outside [0.5, 8] s")
        if int(self.n_freq) < 0 or int(self.haz_hidden) < 8 or int(self.ep_hidden) < 8:
            raise ValueError(f"v4: bad sizes n_freq {self.n_freq} haz_hidden {self.haz_hidden} "
                             f"ep_hidden {self.ep_hidden}")
        if float(self.prior_n_eff) < 1.0:
            raise ValueError(f"v4.prior_n_eff {self.prior_n_eff} < 1")
        if self.version != CONFIG_VERSION:
            raise ValueError(f"v4.version {self.version!r} != {CONFIG_VERSION!r}")
        return self

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "V4Cfg":
        """None / {} -> off.  Unknown keys -> ValueError.  Bools given as strings ('true' / '0') are parsed."""
        if not d:
            return cls()
        d = dict(d)
        names = {f.name for f in dataclasses.fields(cls)}
        bad = sorted(set(d) - names)
        if bad:
            raise ValueError(f"v4: unknown key(s) {bad}")
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
