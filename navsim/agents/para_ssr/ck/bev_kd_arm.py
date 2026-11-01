"""CK2 e2e BEV-KD arm ("CK2 + simple BEV KD"): the ck/bev_kd.py BEVFeatureKD wired as an agent module (SPEC s1-6, s6).

Used only when ck_e2e2.bev_kd.enabled (default off).  ParaSSRAgent registers BEVKDArm as `ck_bev_kd` (state_dict /
DDP / its own optimiser group); CKE2E2.loss calls it every micro-batch and adds its term to the CK loss (outside the v2
grad balancer).  Teachers (bev_kd.teachers): 'det' = BEVFusion t0 BEV (kd_bev_0 / kd_ok_0, z-score det_run/norm.npz)
and 'map' = ReSMap BEV (kd_bev_1 / kd_ok_1, z-score map_run/norm_map.npz); the arm uses [det, map] (user 2026-10-08
~20:40 KST "BEV KD에 Resmap도 추가해"), [det] alone is the old arm, unchanged bit for bit.

Per-teacher strength (one GradScale branch + one RatioController per teacher; user 2026-10-08: MAP must not be drowned
by DET)
  term = adapter_weight * sum_t L_t(GradScale(bev_embed, lam_t / adapter_weight))
    -> adapter t learns at the fixed weight adapter_weight from its own loss L_t only (a zero-init adapter fits while
       lam_t = 0; the number of teachers does not change an adapter's step);
       the student BEV receives sum_t lam_t * dL_t / d bev_embed (GradScale: identity forward, grad * s backward).
  lam_t = RatioController_t.weight(mb): 0 before start_mb (10600 = the v2 balancer warm-up) or before a valid
        measurement of teacher t, else min(ratio_t * exp(EMA log(g_plan / g_t)), cap_t * (g_plan / g_t)_last, w_max)
        i.e. lam_t * ||dL_t / dBEV|| ~ ratio_t (0.1) x ||dL_plan / dBEV|| on average and <= cap_t (0.25) x at the last
        measurement, for EVERY teacher separately (ratio_t / cap_t = bev_kd.ratio_<t> / cap_<t> if set, else
        bev_kd.ratio / cap).  The cap holds at the measurement only, not at every step in between (report 48 s6).
  Measurement (CKE2E2._bev_kd_term): on every micro-batch whose v2 loss logged 'gnorm/plan' (the v2 grad-norm /
        balancer micro-batches, identical on all ranks), g_t = ||dL_t / dx_t|| (x_t = teacher t's GradScale output,
        unit weight) for every teacher, ONE all-reduce of the vector (mean over the ranks), g_plan = logs['gnorm/plan']
        (already all-reduced by the v2 loss); the new lam_t apply from the next micro-batch.  Every controller input is
        identical on all ranks, so the controllers are identical on all ranks and the rank-0 copy that Lightning saves
        in the checkpoint (CKE2E2Callback state 'bev_ctrl' = {t: state}) restores every rank exactly.
Tokens without a teacher BEV (kd_ok_<i> False, e.g. no ReSMap BEV) are masked out of that teacher's loss only and
counted (n_ok/<t>; CKE2E2 logs bevkd/<t>/ok_frac and cumulates bevkd_miss_<t>).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn

from .bev_kd import TEACHERS, BEVFeatureKD, norm_from_run, teacher_inputs, teacher_keys


class GradScale(torch.autograd.Function):
    """Identity forward; backward multiplies the incoming gradient by the python float s (s may be 0)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, s: float) -> torch.Tensor:
        ctx.s = float(s)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        return g * ctx.s, None


class RatioController:
    """lam from periodic gradient measurements (SPEC NU31 / NU33).

    update(g_kd_unit, g_ref): skipped (n_skip) if non-finite, g_kd_unit <= min_g or g_ref <= 0; else r = g_ref /
      g_kd_unit, log_r = first valid log r, later m * log_r + (1 - m) * log r; r_last = r.
    weight(mb) = 0 if mb < start_mb or no valid measurement, else min(ratio * exp(log_r), cap * r_last, w_max)."""

    def __init__(self, ratio: float = 0.1, cap: float = 0.25, m: float = 0.9, w_max: Optional[float] = None,
                 start_mb: int = 10600, min_g: float = 1e-12):
        if not (ratio > 0 and cap > 0 and 0.0 < m < 1.0):
            raise ValueError(f"RatioController needs ratio, cap > 0 and 0 < m < 1 (got {ratio}, {cap}, {m})")
        if w_max is not None and not w_max > 0:
            raise ValueError("w_max must be > 0 or None")
        self.ratio, self.cap, self.m = float(ratio), float(cap), float(m)
        self.w_max = None if w_max is None else float(w_max)
        self.start_mb, self.min_g = int(start_mb), float(min_g)
        self.log_r: Optional[float] = None
        self.r_last: Optional[float] = None
        self.n = 0
        self.n_skip = 0

    def update(self, g_kd_unit: float, g_ref: float) -> None:
        g, ref = float(g_kd_unit), float(g_ref)
        if not (math.isfinite(g) and math.isfinite(ref)) or g <= self.min_g or ref <= 0.0:
            self.n_skip += 1
            return
        r = ref / g
        lr = math.log(r)
        self.log_r = lr if self.log_r is None else self.m * self.log_r + (1.0 - self.m) * lr
        self.r_last = r
        self.n += 1

    def weight(self, mb: int) -> float:
        if int(mb) < self.start_mb or self.log_r is None or self.r_last is None:
            return 0.0
        w = min(self.ratio * math.exp(self.log_r), self.cap * self.r_last)
        if self.w_max is not None:
            w = min(w, self.w_max)
        return float(w)

    def state_dict(self) -> Dict[str, Any]:
        return {"log_r": self.log_r, "r_last": self.r_last, "n": int(self.n), "n_skip": int(self.n_skip)}

    def load_state_dict(self, sd: Mapping[str, Any]) -> None:
        self.log_r = None if sd.get("log_r") is None else float(sd["log_r"])
        self.r_last = None if sd.get("r_last") is None else float(sd["r_last"])
        self.n, self.n_skip = int(sd.get("n", 0)), int(sd.get("n_skip", 0))


class TeacherRatioControllers:
    """One RatioController per BEV-KD teacher (same m / w_max / start_mb; ratio and cap per teacher, default shared).

    weights(mb) -> {t: lam_t};  update({t: g_t}, g_ref) updates every teacher's controller with its own unit BEV
    gradient norm against the same reference (each skips on its own rules);  state_dict() -> {t: RatioController
    state};  load_state_dict also takes the old single-controller state (keys log_r / r_last / n / n_skip) of a
    one-teacher arm.  A state for a different teacher set is an error (a resumed run must keep its teachers)."""

    def __init__(self, teachers: Sequence[str], ratio: float = 0.1, cap: float = 0.25, m: float = 0.9,
                 w_max: Optional[float] = None, start_mb: int = 10600, min_g: float = 1e-12,
                 per_teacher: Optional[Mapping[str, Mapping[str, Optional[float]]]] = None):
        self.teachers = tuple(str(t) for t in teachers)
        if not self.teachers or len(set(self.teachers)) != len(self.teachers) or \
                any(t not in TEACHERS for t in self.teachers):
            raise ValueError(f"teachers must be a non-empty subset of {TEACHERS}, got {list(teachers)}")
        pt = dict(per_teacher or {})
        bad = sorted(set(pt) - set(self.teachers))
        if bad:
            raise ValueError(f"per-teacher overrides for teachers not in use: {bad}")
        self.ctrl: Dict[str, RatioController] = {}
        for t in self.teachers:
            o = {k: v for k, v in dict(pt.get(t) or {}).items() if v is not None}
            if set(o) - {"ratio", "cap"}:
                raise ValueError(f"per-teacher override keys must be ratio / cap, got {sorted(o)}")
            self.ctrl[t] = RatioController(ratio=o.get("ratio", ratio), cap=o.get("cap", cap), m=m, w_max=w_max,
                                           start_mb=start_mb, min_g=min_g)

    def __getitem__(self, t: str) -> RatioController:
        return self.ctrl[t]

    def weights(self, mb: int) -> Dict[str, float]:
        return {t: self.ctrl[t].weight(mb) for t in self.teachers}

    def update(self, g_units: Mapping[str, float], g_ref: float) -> None:
        missing = [t for t in self.teachers if t not in g_units]
        if missing:
            raise ValueError(f"no unit gradient norm for teacher(s) {missing}")
        for t in self.teachers:
            self.ctrl[t].update(g_units[t], g_ref)

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        return {t: self.ctrl[t].state_dict() for t in self.teachers}

    def load_state_dict(self, sd: Mapping[str, Any]) -> None:
        sd = dict(sd)
        if "log_r" in sd:                              # old single RatioController (teachers [det] only)
            if len(self.teachers) != 1:
                raise ValueError(f"single-controller BEV-KD state cannot be loaded into teachers {list(self.teachers)}")
            self.ctrl[self.teachers[0]].load_state_dict(sd)
            return
        if set(sd) != set(self.teachers):
            raise ValueError(f"BEV-KD controller state for teachers {sorted(sd)} != teachers in use "
                             f"{sorted(self.teachers)} (a resumed run must keep bev_kd.teachers)")
        for t in self.teachers:
            self.ctrl[t].load_state_dict(sd[t])


def controllers_from_cfg(b: Mapping[str, Any], teachers: Optional[Sequence[str]] = None) -> TeacherRatioControllers:
    """TeacherRatioControllers from a (coerced) ck_e2e2.bev_kd dict; ratio_<t> / cap_<t> (None = shared value)."""
    teachers = tuple(b["teachers"] if teachers is None else teachers)
    per = {t: {"ratio": b.get(f"ratio_{t}"), "cap": b.get(f"cap_{t}")} for t in teachers}
    return TeacherRatioControllers(teachers, ratio=b["ratio"], cap=b["cap"], m=b["m"], w_max=b["w_max"],
                                   start_mb=b["start_mb"], per_teacher=per)


class BEVKDArm(nn.Module):
    """BEVFeatureKD (adapters 'kd.adapters.<t>.proj.*' + frozen teacher z-score buffers 'kd.zscore.<t>.*') on the
    teachers in cfg['teachers'] (target keys from teacher_keys(['T', 'M']): det -> kd_bev_0 / kd_ok_0, map ->
    kd_bev_1 / kd_ok_1 = the CK2E2ETargetBuilder teacher-run order (det, map)).  z-score statistics from the CK2
    teacher runs (det_run/norm.npz, map_run/norm_map.npz; bev_kd.NORM_FILE).  One GradScale input per teacher, so
    each teacher's BEV gradient has its own weight lam_t (module docstring)."""

    def __init__(self, cfg_bev_kd: Mapping[str, Any], det_run: str, map_run: str):
        super().__init__()
        c = dict(cfg_bev_kd)
        self.teachers = tuple(str(t) for t in c["teachers"])
        runs = {"det": det_run, "map": map_run}
        missing = [t for t in self.teachers if not runs.get(t)]
        if missing:
            raise ValueError(f"BEV KD teacher(s) {missing}: no teacher run for the z-score statistics")
        norms = {t: norm_from_run(runs[t], t) for t in self.teachers}
        self.kd = BEVFeatureKD(self.teachers, norms, distance=str(c["distance"]), target_clip=c.get("target_clip"),
                               init=str(c["init"]), ln=bool(c["ln"]))
        self.keys = teacher_keys(["T", "M"])
        self.adapter_weight = float(c["adapter_weight"])
        if not self.adapter_weight > 0:
            raise ValueError("bev_kd.adapter_weight must be > 0")
        self.runs = {t: str(runs[t]) for t in self.teachers}

    def forward(self, bev_embed: torch.Tensor, targets: Mapping[str, torch.Tensor],
                lam: Union[float, Mapping[str, float]]) -> Dict[str, torch.Tensor]:
        """lam: {t: lam_t} for every teacher (or one float for all).  -> BEVFeatureKD outputs (loss = mean_t L_t
        unweighted, loss/<t> with graph, raw/<t>, n_ok/<t>, ok_frac/<t>, cos/<t>, fve/<t>, w_dev/<t>) + per teacher
        x/<t> (its GradScale output: the unit-weight gradient point), lam/<t>, term/<t> = adapter_weight * L_t
        (BEV gradient lam_t * dL_t / d bev_embed) and term = sum_t term/<t>."""
        lams = {t: float(lam[t] if isinstance(lam, Mapping) else lam) for t in self.teachers}
        xs = {t: GradScale.apply(bev_embed, lams[t] / self.adapter_weight) for t in self.teachers}
        bev, ok = teacher_inputs(targets, self.keys, self.teachers)
        out = self.kd(xs, bev, ok)
        term = None
        for t in self.teachers:
            out[f"x/{t}"] = xs[t]
            out[f"lam/{t}"] = torch.tensor(lams[t], device=bev_embed.device)
            out[f"term/{t}"] = self.adapter_weight * out[f"loss/{t}"]
            term = out[f"term/{t}"] if term is None else term + out[f"term/{t}"]
        out["term"] = term
        return out

    @staticmethod
    def unit_grad_norms(out: Mapping[str, torch.Tensor], teachers: Sequence[str]) -> Dict[str, float]:
        """{t: ||d L_t / d x_t||} (unit-weight BEV gradient of each teacher's own loss; autograd.grad with
        retain_graph: no .grad is touched).  0 for a teacher whose loss has no graph to x_t (no ok token, eval)."""
        res: Dict[str, float] = {}
        for t in teachers:
            loss, x = out[f"loss/{t}"], out[f"x/{t}"]
            g = None
            if isinstance(loss, torch.Tensor) and loss.requires_grad and x.requires_grad:
                g = torch.autograd.grad(loss, x, retain_graph=True, allow_unused=True)[0]
            res[t] = 0.0 if g is None else float(g.float().norm())
        return res


def build_bev_kd_arm(ck2cfg) -> BEVKDArm:
    """BEVKDArm from a CKE2E2Config (or anything CKE2E2Config.from_any accepts)."""
    from .online2 import CKE2E2Config

    c = CKE2E2Config.from_any(ck2cfg)
    return BEVKDArm(c.bev_kd, c.teacher_det_run, c.teacher_map_run)


def bev_kd_parameters(agent) -> List[nn.Parameter]:
    """Trainable BEV-KD adapter parameters of the agent (empty without the arm)."""
    m = getattr(agent, "ck_bev_kd", None)
    return [] if m is None else [p for p in m.parameters() if p.requires_grad]
