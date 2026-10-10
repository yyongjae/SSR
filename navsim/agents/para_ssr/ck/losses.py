"""CK losses (report 44 §4).

  score_bce      GT label BCE-with-logits (soft targets allowed, e.g. nc 0.5), mean over ok candidates and 5 keys.
  score_kd_bce   score KD to the teacher probability: soft BCE minus the target entropy (= Bernoulli KL).  Same
                 gradient as the plain soft BCE; 0 when the student equals the teacher.
  ctrl_kd_l1     correction KD in the decoded control space: w_lon * mean|c_lon_s - c_lon_t| + w_lat * mean|e_lat_s -
                 e_lat_t| (6 + 6 controls, c_lon[2:] m/s, e_lat[2:] m; default lon 0.25 / lat 1.0).
  correction_loss v1 GT surrogate (C_col, C_ttc, C_dac, L_prog, C_cmf, C_mod; R_T4 / E2 weights and margins) on the
                 decoded corrections of the K candidates of the GT-ok tokens (train_refiner.surrogate_terms_batch);
                 any non-finite weighted term drops the micro-batch surrogate (E2 rule).
  EmaBalancer    E2 'ema' KD weight w = min(cap, ratio * EMA_hat(L_sur_w) / max(EMA_hat(L_KD), floor)), 0 before
                 start_step (single process; same math as refiner.e2e.StageE.ema_*).
  lead_bce       optional lead-vehicle-deceleration auxiliary BCE (has_lead == 1 and D_1 finite).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ..refiner.e2e import train_refiner_module
from .constants import CK_KEYS, KD_CTRL_W, N_CTRL, SUR_MARGINS, SUR_TERMS, SUR_WEIGHTS


# ----------------------------------------------------------------------------------------------- helpers
def _mask(ok: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ok [T, K] AND all target values finite -> bool [T, K]."""
    return ok.bool().to(y.device) & torch.isfinite(y).all(-1)


def _key_mean(per: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """per [T, K, C], m [T, K] bool -> [C] mean over masked candidates (0 without any)."""
    mf = m.to(per.dtype)[..., None]
    return (per * mf).sum((0, 1)) / torch.clamp(mf.sum(), min=1.0)


# ----------------------------------------------------------------------------------------------- scores
def score_bce(logit: torch.Tensor, y: torch.Tensor, ok: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    """logit [T, K, 5], y [T, K, 5] in [0, 1] (CK_KEYS order), ok [T, K] -> (mean over ok candidates and keys,
    per-key floats).  Non-finite targets count as not ok."""
    m = _mask(ok, y)
    yy = torch.where(m[..., None], y.float(), torch.zeros_like(y, dtype=torch.float32)).clamp(0.0, 1.0)
    per = F.binary_cross_entropy_with_logits(logit.float(), yy, reduction="none")
    pk = _key_mean(per, m)
    return pk.mean(), {k: float(pk[i].detach()) for i, k in enumerate(CK_KEYS)}


def bernoulli_entropy(p: torch.Tensor) -> torch.Tensor:
    return -(torch.special.xlogy(p, p) + torch.special.xlogy(1.0 - p, 1.0 - p))


def score_kd_bce(logit: torch.Tensor, p_t: torch.Tensor, ok: torch.Tensor, entropy_correct: bool = True
                 ) -> torch.Tensor:
    """Score KD: BCE-with-logits(logit [T, K, 5], teacher probability p_t [T, K, 5]) over ok candidates and keys.
    entropy_correct=True subtracts H(p_t) (a constant: same gradient) so the value is the Bernoulli KL, 0 when the
    student equals the teacher.  p_t is detached."""
    p = p_t.detach().float()
    m = _mask(ok, p)
    pp = torch.where(m[..., None], p, torch.zeros_like(p)).clamp(0.0, 1.0)
    per = F.binary_cross_entropy_with_logits(logit.float(), pp, reduction="none")
    if entropy_correct:
        per = per - bernoulli_entropy(pp)
    return _key_mean(per, m).mean()


# ----------------------------------------------------------------------------------------------- correction KD
def _ctrl6(x: torch.Tensor) -> torch.Tensor:
    return x[..., 2:] if x.shape[-1] == N_CTRL + 2 else x


def ctrl_kd_l1(c_lon_s: torch.Tensor, e_lat_s: torch.Tensor, c_lon_t: torch.Tensor, e_lat_t: torch.Tensor,
               ok: torch.Tensor, w_lon: float = KD_CTRL_W["lon"], w_lat: float = KD_CTRL_W["lat"]
               ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Decoded-control KD: [T, K, 6] (or the full [T, K, 8] decode outputs, sliced to [2:]); targets detached;
    -> (w_lon * lon + w_lat * lat, {'lon': mean |dc_lon|, 'lat': mean |de_lat|}) over ok candidates (and finite
    targets) and the 6 controls of each branch."""
    cs, es = _ctrl6(c_lon_s).float(), _ctrl6(e_lat_s).float()
    ct, et = _ctrl6(c_lon_t).detach().float().to(cs.device), _ctrl6(e_lat_t).detach().float().to(cs.device)
    m = ok.bool().to(cs.device) & torch.isfinite(ct).all(-1) & torch.isfinite(et).all(-1)
    ct = torch.where(m[..., None], ct, cs.detach())
    et = torch.where(m[..., None], et, es.detach())
    lon = _key_mean((cs - ct).abs(), m).mean()
    lat = _key_mean((es - et).abs(), m).mean()
    return w_lon * lon + w_lat * lat, {"lon": float(lon.detach()), "lat": float(lat.detach())}


# ----------------------------------------------------------------------------------------------- surrogate
def surrogate_index(ref_gt_ok: torch.Tensor, human_traj: torch.Tensor) -> torch.Tensor:
    """Tokens usable by the surrogate: GTLoader ref_gt_ok and a finite GT human trajectory -> LongTensor [n]."""
    ok = ref_gt_ok.reshape(-1).bool() & torch.isfinite(human_traj.reshape(human_traj.shape[0], -1)).all(1).to(
        ref_gt_ok.device)
    return ok.nonzero()[:, 0]


def surrogate_batch_k(ref: Dict[str, torch.Tensor], idx: torch.Tensor, tau: torch.Tensor, human_traj: torch.Tensor,
                      v0: torch.Tensor, a0: torch.Tensor) -> Dict[str, torch.Tensor]:
    """K-candidate version of refiner.e2e.surrogate_batch: ref = batched GTLoader outputs (ref_obj_kf, ref_obj_first,
    ref_obj_meta, ref_obj_n, ref_obj_n_kf, ref_obj_R, ref_obj_ego_kf, ref_sdf, ref_cl_xy, ref_cl_valid, ref_cl_n,
    ref_p_pdm, ...), idx [n] token rows, tau [T, K, 8, 3] (the candidates = surrogate drafts), human_traj [T, 8, 3]
    (GT future, for the human-overlap mask), v0 / a0 [T] -> train_refiner.scene_from_batch batch with tau0 [n, K, 8, 3]
    (+ 'idx').  Objects / centerline trimmed to the subset's maxima."""
    g = lambda k: ref[f"ref_{k}"][idx]
    n = int(idx.numel())
    n_obj = int(max(int(g("obj_n").max()), 1)) if n else 1
    n_cl = int(max(min(int(g("cl_n").max()) if n else 2, ref["ref_cl_xy"].shape[1]), 2))
    return {
        "tau0": tau[idx].float(), "human_traj": human_traj[idx].float(),
        "v0": v0[idx].float(), "a0": a0[idx].float(),
        "obj_kf": g("obj_kf")[:, :n_obj].float(), "obj_first": g("obj_first")[:, :n_obj].float(),
        "obj_meta": g("obj_meta")[:, :n_obj].long(),
        "obj_valid": torch.arange(n_obj, device=idx.device)[None] < g("obj_n").long()[:, None],
        "obj_n_kf": g("obj_n_kf").long(), "obj_R": g("obj_R").float(), "obj_ego_kf": g("obj_ego_kf").float(),
        "sdf": g("sdf"), "cl_xy": g("cl_xy")[:, :n_cl].float(), "cl_valid": g("cl_valid")[:, :n_cl].bool(),
        "pdm_progress_eff": g("p_pdm").double(), "idx": idx,
    }


def restrict_dec(dec: Dict, rows: torch.Tensor) -> Dict:
    """Index every [B, ...] tensor of a decode() dict (and of its flags) by rows; the DraftPath is dropped."""
    B = dec["traj"].shape[0]
    out = {}
    for k, v in dec.items():
        if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == B:
            out[k] = v[rows]
        elif isinstance(v, dict):
            out[k] = {a: (b[rows] if isinstance(b, torch.Tensor) and b.dim() > 0 and b.shape[0] == B else b)
                      for a, b in v.items()}
        elif k != "path":
            out[k] = v
    return out


def correction_loss(dec_raw: Dict, batch: Dict, n: int, K: int, weights: Optional[Dict[str, float]] = None,
                    margins: Optional[Dict[str, float]] = None, valid: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, Dict[str, float], int]:
    """Surrogate loss of the decoded corrections (correct()['raw'], flat over T*K candidates, or already the n*K rows of
    the surrogate tokens) on the surrogate_batch_k batch of n tokens.  weights / margins default SUR_WEIGHTS /
    SUR_MARGINS (always passed explicitly to SurrogateConfig).  valid: optional [n, K] candidate mask.
    -> (sum_k w_k * mean term_k (f32 scalar), term means (floats, incl. P1_minus_P0), nonfinite (1 = the micro-batch
    surrogate was dropped because a weighted term was non-finite))."""
    from ..refiner.surrogate import SurrogateConfig

    w = dict(SUR_WEIGHTS if weights is None else weights)
    mg = dict(SUR_MARGINS if margins is None else margins)
    zero = dec_raw["traj"].new_zeros((), dtype=torch.float32)
    means = {k: 0.0 for k in SUR_TERMS + ("P1_minus_P0",)}
    if n == 0:
        return zero, means, 0
    dec = dec_raw
    if dec_raw["traj"].shape[0] != n * K:
        if "idx" not in batch:
            raise ValueError("correction_loss: dec_raw is not restricted to the batch tokens and batch has no 'idx'")
        idx = batch["idx"].to(dec_raw["traj"].device)
        rows = (idx[:, None] * K + torch.arange(K, device=idx.device)[None]).reshape(-1)
        dec = restrict_dec(dec_raw, rows)
    terms = train_refiner_module().surrogate_terms_batch(dec, batch, n, K, SurrogateConfig(**mg),
                                                         ttc_grad=bool(w.get("ttc", 0.0)))
    m = (torch.ones(n * K, device=dec["traj"].device) if valid is None
         else valid.reshape(-1).to(dec["traj"].device).float())
    den = torch.clamp(m.sum(), min=1.0)
    bad = any(not bool(torch.isfinite(terms[k]).all()) for k in SUR_TERMS if w.get(k, 0.0))
    loss = zero
    for k in SUR_TERMS:
        v = torch.nan_to_num(terms[k].float(), nan=0.0, posinf=1e3, neginf=-1e3)
        mv = (v * m).sum() / den
        means[k] = float(mv.detach())
        if w.get(k, 0.0):
            loss = loss + float(w[k]) * mv
    means["P1_minus_P0"] = float((((terms["P1"] - terms["P0"]).float() * m).sum() / den).detach())
    if bad or not bool(torch.isfinite(loss)):
        return zero, means, 1
    return loss, means, 0


# ----------------------------------------------------------------------------------------------- KD balance
class EmaBalancer:
    """E2 'ema' KD weight (refiner.e2e.StageE.ema_update / ema_hat / ema_weight, single process):
    update(l_sur_weighted, l_kd) with EMA momentum m (non-finite values skip the update), bias-corrected;
    weight(step) = 0 before start_step (or before the first update), else min(cap, ratio * EMA_hat(L_sur_w) /
    max(EMA_hat(L_KD), floor)) (cap None = no cap)."""

    def __init__(self, ratio: float = 1.0, m: float = 0.99, floor: float = 1e-4, cap: Optional[float] = 10.0,
                 start_step: int = 500):
        self.ratio, self.m, self.floor = float(ratio), float(m), float(floor)
        self.cap = None if cap is None else float(cap)
        self.start_step = int(start_step)
        self.sur, self.kd, self.n = 0.0, 0.0, 0

    def update(self, l_sur_weighted: float, l_kd: float) -> None:
        s, k = float(l_sur_weighted), float(l_kd)
        if not (math.isfinite(s) and math.isfinite(k)):
            return
        self.sur = self.m * self.sur + (1.0 - self.m) * s
        self.kd = self.m * self.kd + (1.0 - self.m) * k
        self.n += 1

    def hat(self) -> Tuple[float, float]:
        if self.n == 0:
            return 0.0, 0.0
        c = 1.0 - self.m ** self.n
        return self.sur / c, self.kd / c

    def weight(self, step: int, capped: bool = True) -> float:
        if self.n == 0 or int(step) < self.start_step:
            return 0.0
        s, k = self.hat()
        w = self.ratio * s / max(k, self.floor)
        if capped and self.cap is not None:
            w = min(w, self.cap)
        return float(w)

    def state_dict(self) -> Dict:
        return {"sur": self.sur, "kd": self.kd, "n": self.n}

    def load_state_dict(self, sd: Dict) -> None:
        self.sur, self.kd, self.n = float(sd["sur"]), float(sd["kd"]), int(sd["n"])


# ----------------------------------------------------------------------------------------------- lead aux
def lead_bce(logit: torch.Tensor, has_lead: torch.Tensor, d1: torch.Tensor) -> torch.Tensor:
    """Lead-vehicle-decelerates-within-4-s auxiliary BCE: logit [T], has_lead [T] (0/1), d1 [T] (0/1, NaN =
    censored) -> mean over tokens with has_lead == 1 and finite d1 (0 without any)."""
    y = d1.float().to(logit.device)
    m = (has_lead.float().to(logit.device) > 0.5) & torch.isfinite(y)
    per = F.binary_cross_entropy_with_logits(logit.float(), torch.where(m, y, torch.zeros_like(y)).clamp(0, 1),
                                             reduction="none")
    mf = m.float()
    return (per * mf).sum() / torch.clamp(mf.sum(), min=1.0)
