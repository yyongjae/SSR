"""CK2 simple BEV feature KD (e2e arm "CK2 + BEV KD"; wired into the e2e trainer by ck/bev_kd_arm.py).

Structure, per teacher t in TEACHERS = ("det", "map") that is switched on
    student bev_embed [B, 5000, 256]  (v2 BEVFormer output; index = row * 100 + col = the S grid:
                                       row r -> x = (r + .5) * 0.64 m forward, col c -> y_left = 32 - (c + .5) * 0.64 m,
                                       origin rear axle; refiner/adapters.py, report refiner_T/stageE_impl_plan.md s1)
      -> per-cell LayerNorm over the 256 channels, no affine (bev_embed is already a post-norm output, per-cell std
         0.99; this only pins the scale so the BEV cannot lower the loss by shrinking)
      -> 1x1 adapter A_t = Linear(256, 256) applied per cell (== 1x1 conv; 65,792 parameters per teacher)
    teacher BEV  targets kd_bev_i f16 [B, 256, 50, 100], already on the S grid (refiner.e2e.GTLoader:
                 BEVFusion t0 bev_feature lateral flip, ReSMap neck bev transpose), ok flag kd_ok_i
      -> per-channel z-score with the frozen statistics of the CK teacher run (norm.npz = DET, norm_map.npz = MAP),
         buffers, not parameters
      -> optional clip |z| <= target_clip (BEVFusion is post-ReLU and heavy-tailed: z up to ~30)
    per cell u: d(u) = mean_c (A_t(LN s)(u) - z_t(u))^2           distance 'mse'
                d(u) = 1 - cos_c(A_t(LN s)(u), z_t(u))             distance 'cosine'
    per token:  mean_u w(u) d(u)  (w = 1, or a passed cell weight normalised to mean 1 per token)
    L_t = mean over ok tokens; L = mean over the teachers in use (so one or two teachers give the same scale).
Gradient flows into A_t and bev_embed (hence the BEV encoder and backbone); nothing flows to the teacher side.

Adapter init (init=...)
  'zero'     : W = 0, b = 0.  At step 0 the output is 0 and the BEV gets NO gradient (dL/ds = W^T r = 0); W first
               learns the linear map from LN(s) to the teacher, and the BEV gradient grows with it, only along
               directions W already uses.  No arbitrary channel pairing is forced on the student.  (default)
  'identity' : W = I, b = 0 (readout/distill.py kd_adapter, kyungmin v1 kd_feature).  Pushes student channel i towards
               teacher channel i before W has learned anything.
  'default'  : torch's Linear init under a fixed seed.

Weight
  The module returns the UNWEIGHTED loss (mean over the teachers) and every teacher's own unweighted loss loss/<t> (the
  e2e arm, ck/bev_kd_arm.py, weights each teacher separately: one GradScale input + one RatioController per teacher).
  A caller may also multiply the mean by one weight: a fixed lambda, or ShareController,
  which sets lambda so that lambda * g_kd / (g_ref + lambda * g_kd) = target share, with g = ||dL / d bev_embed||
  measured by bev_grad_norms (g_kd for the unweighted KD loss, g_ref for the rest of the loss, e.g. v2 after its
  balancer (+ CK)).  Same quantity as ck/online.py's ck/gshare_ck and modules/grad_balance.py.

Not here (ck/bev_kd_arm.py, ck/online2.py, para_ssr_agent.py): registering the module on the agent (DDP + optimizer),
the per-teacher lambda schedule, logging.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BEV_H, BEV_W, BEV_C = 50, 100, 256
N_CELLS = BEV_H * BEV_W
TEACHERS = ("det", "map")
ARM_OF = {"det": "T", "map": "M"}                    # CK teacher run arm (config.json 'arm')
NORM_FILE = {"det": "norm.npz", "map": "norm_map.npz"}
DISTANCES = ("mse", "cosine")
INITS = ("zero", "identity", "default")
INIT_SEED = 7_401


# ----------------------------------------------------------------------------------------------- layout helpers
def sgrid_to_tokens(x: torch.Tensor) -> torch.Tensor:
    """S grid [B, C, 50, 100] -> [B, 5000, C] (index row * 100 + col, the bev_embed layout)."""
    if x.dim() != 4 or tuple(x.shape[2:]) != (BEV_H, BEV_W):
        raise ValueError(f"expected [B, C, {BEV_H}, {BEV_W}], got {tuple(x.shape)}")
    return x.flatten(2).transpose(1, 2)


def tokens_to_sgrid(x: torch.Tensor) -> torch.Tensor:
    """[B, 5000, C] -> S grid [B, C, 50, 100] (inverse of sgrid_to_tokens; = ck.online.bev_sgrid)."""
    if x.dim() != 3 or x.shape[1] != N_CELLS:
        raise ValueError(f"expected [B, {N_CELLS}, C], got {tuple(x.shape)}")
    return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], BEV_H, BEV_W)


def teacher_keys(teacher_arms: Sequence[str]) -> Dict[str, Tuple[str, str]]:
    """GTLoader target keys per teacher from the e2e teacher-run arms in loading order (['T', 'M'] ->
    {'det': ('kd_bev_0', 'kd_ok_0'), 'map': ('kd_bev_1', 'kd_ok_1')})."""
    out: Dict[str, Tuple[str, str]] = {}
    for i, arm in enumerate(teacher_arms):
        name = {"T": "det", "M": "map"}.get(arm)
        if name is None:
            raise ValueError(f"teacher arm {arm!r} not in (T, M)")
        if name in out:
            raise ValueError(f"teacher {name} listed twice in {list(teacher_arms)}")
        out[name] = (f"kd_bev_{i}", f"kd_ok_{i}")
    return out


def teacher_inputs(targets: Mapping[str, torch.Tensor], keys: Mapping[str, Tuple[str, str]],
                   teachers: Sequence[str]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """(bev, ok) dicts for `teachers` out of an e2e targets dict; a missing key raises (a silent zero map would
    train the student towards a constant)."""
    bev, ok = {}, {}
    for t in teachers:
        if t not in keys:
            raise KeyError(f"BEV KD teacher {t!r} has no target keys (teacher runs loaded: {sorted(keys)})")
        kb, ko = keys[t]
        bev[t], ok[t] = targets[kb], targets[ko]
    return bev, ok


def load_teacher_norm(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """(mean [256], std [256]) float32 of a CK / v1 teacher run's norm.npz / norm_map.npz."""
    with np.load(path, allow_pickle=False) as z:
        mean, std = z["mean"].astype(np.float32), z["std"].astype(np.float32)
    if mean.shape != (BEV_C,) or std.shape != (BEV_C,) or not (np.all(np.isfinite(mean)) and np.all(std > 0)):
        raise ValueError(f"{path}: bad z-score statistics {mean.shape} {std.shape}")
    return mean, std


def norm_from_run(run_dir: Union[str, Path], teacher: str) -> Tuple[np.ndarray, np.ndarray]:
    return load_teacher_norm(Path(run_dir) / NORM_FILE[teacher])


# ----------------------------------------------------------------------------------------------- modules
class TeacherZ(nn.Module):
    """(x - mean_c) / std_c on [B, 5000, 256] tokens, frozen buffers; optional clip to |z| <= clip."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, clip: Optional[float] = None):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(np.asarray(mean, np.float32)).view(1, 1, BEV_C).clone())
        self.register_buffer("std", torch.as_tensor(np.asarray(std, np.float32)).view(1, 1, BEV_C).clone())
        self.clip = None if clip is None else float(clip)

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        z = (tok.float() - self.mean) / self.std
        return z.clamp(-self.clip, self.clip) if self.clip is not None else z


class BEVKDAdapter(nn.Module):
    """bev_embed [B, 5000, 256] -> per-cell LayerNorm (no affine, optional) -> Linear 256->256 per cell."""

    def __init__(self, init: str = "zero", ln: bool = True, seed: int = INIT_SEED):
        super().__init__()
        if init not in INITS:
            raise ValueError(f"init must be one of {INITS}, got {init!r}")
        self.ln = bool(ln)
        with torch.random.fork_rng(devices=[]):          # never consumes the global RNG (arms stay aligned)
            torch.manual_seed(int(seed))
            self.proj = nn.Linear(BEV_C, BEV_C)
        with torch.no_grad():
            if init == "zero":
                self.proj.weight.zero_()
                self.proj.bias.zero_()
            elif init == "identity":
                self.proj.weight.copy_(torch.eye(BEV_C))
                self.proj.bias.zero_()

    def normed(self, bev: torch.Tensor) -> torch.Tensor:
        x = bev.float()
        return F.layer_norm(x, (BEV_C,)) if self.ln else x

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        return self.proj(self.normed(bev))


def _cell_distance(p: torch.Tensor, z: torch.Tensor, distance: str) -> torch.Tensor:
    """[B, 5000, C] x 2 -> [B, 5000]."""
    if distance == "mse":
        return (p - z).pow(2).mean(-1)
    return 1.0 - F.cosine_similarity(p, z, dim=-1, eps=1e-6)


def _cell_weight(w: Optional[torch.Tensor], B: int, like: torch.Tensor) -> Optional[torch.Tensor]:
    if w is None:
        return None
    w = w.to(device=like.device, dtype=torch.float32).reshape(B, -1)
    if w.shape[1] != N_CELLS:
        raise ValueError(f"cell_weight needs {N_CELLS} cells per token, got {w.shape[1]}")
    if bool((w < 0).any()) or not bool(torch.isfinite(w).all()):
        raise ValueError("cell_weight must be finite and >= 0")
    s = w.sum(1, keepdim=True)
    return torch.where(s > 0, w * (N_CELLS / s.clamp(min=1e-12)), torch.ones_like(w))


class BEVFeatureKD(nn.Module):
    """Simple BEV feature KD: one 1x1 adapter + frozen z-score per teacher (see module docstring).

    forward(bev_embed [B, 5000, 256] or {t: [B, 5000, 256]} (one input point per teacher, e.g. per-teacher GradScale
            outputs of the same tensor; bev_kd_arm), teacher_bev {t: [B, 256, 50, 100] (any float dtype)},
            teacher_ok {t: [B] bool}, cell_weight [B, 50, 100] | [B, 5000] | None) -> dict
      loss            scalar, UNWEIGHTED (mean over the teachers in use of the mean over ok tokens); graph to the adapters
                      and bev_embed.  With no ok token anywhere it is 0 * (adapter parameters), so every adapter
                      parameter stays in the graph (DDP) and gets a zero gradient.
      loss/<t>        the teacher's own UNWEIGHTED loss L_t (mean over its ok tokens) WITH its graph (to adapter t and
                      that teacher's bev_embed input only); 0 * (adapter t parameters) when it has no ok token.
                      loss == mean_t loss/<t>.
      raw/<t>         detached per-teacher loss (0 if no ok token)
      ok_frac/<t>     share of ok tokens;  n_ok/<t>  number of ok tokens (int64 scalar; the rest are masked: no
                      teacher BEV for them, e.g. kd_ok_1 False = no ReSMap BEV)
      cos/<t>         mean cell cosine(adapter output, z teacher) over ok tokens (diagnostic)
      fve/<t>         1 - sum (p - z)^2 / sum z^2 over ok tokens (fraction of the z-scored teacher energy explained;
                      the teacher's z has ~zero channel mean, so this is ~R^2)
      w_dev/<t>       ||W - W_init|| / ||I|| (how far the adapter moved; 16 = ||I||_F)
    Fixed per instance: teachers, distance, target_clip, ln, init.  The whole computation runs in float32 with autocast
    off (v2 trains in fp32 anyway because of its grad balancer).
    """

    def __init__(self, teachers: Sequence[str], norms: Mapping[str, Tuple[np.ndarray, np.ndarray]],
                 distance: str = "mse", target_clip: Optional[float] = None, init: str = "zero", ln: bool = True,
                 seed: int = INIT_SEED):
        super().__init__()
        teachers = tuple(teachers)
        if not teachers or any(t not in TEACHERS for t in teachers) or len(set(teachers)) != len(teachers):
            raise ValueError(f"teachers must be a non-empty subset of {TEACHERS}, got {teachers}")
        if distance not in DISTANCES:
            raise ValueError(f"distance must be one of {DISTANCES}, got {distance!r}")
        if target_clip is not None and not float(target_clip) > 0:
            raise ValueError("target_clip must be > 0 or None")
        missing = [t for t in teachers if t not in norms]
        if missing:
            raise ValueError(f"no z-score statistics for {missing}")
        self.teachers, self.distance, self.init = teachers, distance, init
        self.target_clip = None if target_clip is None else float(target_clip)
        self.adapters = nn.ModuleDict({t: BEVKDAdapter(init, ln, seed + 101 * i) for i, t in enumerate(teachers)})
        self.zscore = nn.ModuleDict({t: TeacherZ(*norms[t], clip=target_clip) for t in teachers})
        for t in teachers:
            self.register_buffer(f"w_init_{t}", self.adapters[t].proj.weight.detach().clone(), persistent=False)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config(self) -> Dict:
        return {"teachers": list(self.teachers), "distance": self.distance, "target_clip": self.target_clip,
                "init": self.init, "ln": bool(next(iter(self.adapters.values())).ln), "params": self.param_count()}

    def forward(self, bev_embed: Union[torch.Tensor, Mapping[str, torch.Tensor]],
                teacher_bev: Mapping[str, torch.Tensor], teacher_ok: Mapping[str, torch.Tensor],
                cell_weight: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        per_t = isinstance(bev_embed, Mapping)
        inputs = {t: (bev_embed[t] if per_t else bev_embed) for t in self.teachers}
        for t, x in inputs.items():
            if x.dim() != 3 or tuple(x.shape[1:]) != (N_CELLS, BEV_C):
                raise ValueError(f"bev_embed{'[' + t + ']' if per_t else ''} must be [B, {N_CELLS}, {BEV_C}], got "
                                 f"{tuple(x.shape)}")
        first = inputs[self.teachers[0]]
        B, dev = first.shape[0], first.device
        if any(x.shape[0] != B for x in inputs.values()):
            raise ValueError("per-teacher bev_embed inputs must share the batch size")
        out: Dict[str, torch.Tensor] = {}
        terms = []
        with torch.autocast(device_type=dev.type, enabled=False):
            w = _cell_weight(cell_weight, B, first)
            for t in self.teachers:
                adapter = self.adapters[t]
                bev_t = inputs[t]
                ok = teacher_ok[t].to(dev).reshape(-1).bool()
                if ok.numel() != B:
                    raise ValueError(f"teacher_ok[{t}] has {ok.numel()} entries for batch {B}")
                tb = teacher_bev[t]
                if tuple(tb.shape) != (B, BEV_C, BEV_H, BEV_W):
                    raise ValueError(f"teacher_bev[{t}] must be [B, {BEV_C}, {BEV_H}, {BEV_W}], got {tuple(tb.shape)}")
                n_ok = int(ok.sum())
                out[f"ok_frac/{t}"] = torch.tensor(n_ok / max(B, 1), device=dev)
                out[f"n_ok/{t}"] = torch.tensor(n_ok, dtype=torch.int64, device=dev)
                with torch.no_grad():
                    out[f"w_dev/{t}"] = (adapter.proj.weight - getattr(self, f"w_init_{t}")).norm() / math.sqrt(BEV_C)
                if n_ok == 0:
                    zero = sum(p.sum() for p in adapter.parameters()) * 0.0
                    terms.append(zero)
                    out[f"loss/{t}"] = zero
                    for k in ("raw", "cos", "fve"):
                        out[f"{k}/{t}"] = torch.zeros((), device=dev)
                    continue
                idx = ok.nonzero(as_tuple=True)[0]
                s = bev_t.index_select(0, idx)
                with torch.no_grad():
                    z = self.zscore[t](sgrid_to_tokens(tb.to(dev).index_select(0, idx)))
                p = adapter(s)
                d = _cell_distance(p, z, self.distance)                                  # [n_ok, 5000]
                per_tok = (d * w.index_select(0, idx)).mean(1) if w is not None else d.mean(1)
                lt = per_tok.mean()
                terms.append(lt)
                out[f"loss/{t}"] = lt
                with torch.no_grad():
                    out[f"raw/{t}"] = lt.detach()
                    out[f"cos/{t}"] = F.cosine_similarity(p, z, dim=-1, eps=1e-6).mean()
                    out[f"fve/{t}"] = 1.0 - (p - z).pow(2).sum() / z.pow(2).sum().clamp(min=1e-12)
            loss = torch.stack(terms).mean()
        out["loss"] = loss
        return out


# ----------------------------------------------------------------------------------------------- gradient share
def bev_grad_norms(losses: Mapping[str, Optional[torch.Tensor]], bev: torch.Tensor) -> Dict[str, float]:
    """||d L_k / d bev|| for each loss (autograd.grad with retain_graph: no .grad is touched, the graph survives for
    the real backward).  A loss that is None / has no graph to bev gives 0."""
    out: Dict[str, float] = {}
    for k, l in losses.items():
        g = None
        if isinstance(l, torch.Tensor) and l.requires_grad and bev.requires_grad:
            g = torch.autograd.grad(l, bev, retain_graph=True, allow_unused=True)[0]
        out[k] = 0.0 if g is None else float(g.float().norm())
    return out


def share_of(weight: float, g_kd_unit: float, g_ref: float) -> float:
    """BEV gradient share of the weighted KD term: w g_kd / (g_ref + w g_kd) (magnitudes, as gshare_ck)."""
    a = float(weight) * float(g_kd_unit)
    return a / max(a + float(g_ref), 1e-30)


def weight_for_share(share: float, g_kd_unit: float, g_ref: float) -> float:
    """lambda with share_of(lambda, g_kd_unit, g_ref) == share."""
    if not 0.0 < share < 1.0:
        raise ValueError("share must be in (0, 1)")
    if not (g_kd_unit > 0 and math.isfinite(g_kd_unit) and math.isfinite(g_ref)):
        raise ValueError("g_kd_unit must be finite and > 0, g_ref finite")
    return share / (1.0 - share) * float(g_ref) / float(g_kd_unit)


class ShareController:
    """lambda(t) for a target BEV gradient share, from periodic measurements (rank-local unless the caller averages
    g_kd_unit / g_ref over ranks before update()).

    update(g_kd_unit, g_ref): r = g_ref / g_kd_unit; the first valid measurement is adopted, later ones enter an EMA of
    log r (momentum m); lambda = clip(share / (1 - share) * exp(EMA log r), w_min, w_max).  A measurement with
    g_kd_unit <= min_g or a non-finite value is skipped (e.g. step 0 of a zero-init adapter, where the KD loss has no
    BEV gradient yet).  Before the first valid measurement lambda = w_init.
    """

    def __init__(self, share: float, w_init: float, m: float = 0.9, w_min: float = 0.0,
                 w_max: float = float("inf"), min_g: float = 1e-12):
        if not 0.0 < share < 1.0:
            raise ValueError("share must be in (0, 1)")
        if not 0.0 <= m < 1.0:
            raise ValueError("m must be in [0, 1)")
        if not 0.0 <= w_min <= w_max:
            raise ValueError("need 0 <= w_min <= w_max")
        self.share, self.m, self.w_min, self.w_max, self.min_g = float(share), float(m), float(w_min), \
            float(w_max), float(min_g)
        self.w_init = float(w_init)
        self.log_r: Optional[float] = None
        self.n = 0
        self.n_skip = 0

    @property
    def weight(self) -> float:
        if self.log_r is None:
            return min(max(self.w_init, self.w_min), self.w_max)
        w = self.share / (1.0 - self.share) * math.exp(self.log_r)
        return float(min(max(w, self.w_min), self.w_max))

    def update(self, g_kd_unit: float, g_ref: float) -> float:
        g_kd_unit, g_ref = float(g_kd_unit), float(g_ref)
        if not (math.isfinite(g_kd_unit) and math.isfinite(g_ref)) or g_kd_unit <= self.min_g or g_ref <= 0.0:
            self.n_skip += 1
            return self.weight
        lr = math.log(g_ref / g_kd_unit)
        self.log_r = lr if self.log_r is None else self.m * self.log_r + (1.0 - self.m) * lr
        self.n += 1
        return self.weight

    def state_dict(self) -> Dict:
        return {"log_r": self.log_r, "n": self.n, "n_skip": self.n_skip}

    def load_state_dict(self, sd: Mapping) -> None:
        self.log_r = None if sd["log_r"] is None else float(sd["log_r"])
        self.n, self.n_skip = int(sd["n"]), int(sd["n_skip"])
