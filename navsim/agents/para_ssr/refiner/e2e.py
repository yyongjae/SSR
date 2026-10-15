"""Stage E (report/refiner_T/stageE_impl_plan.md; PRESTATED_DECISION_RULE "STAGE E PLAN"): the student refiner inside
PARA-SSR end-to-end training.  Everything here is used ONLY when ParaSSRConfig.refiner_mode != 'off'; with 'off' the
agent never imports this module (E0 bit-identical, tools/refiner/stageE_parity.py).

Arms
  E1 : PARA-SSR + student refiner R_S (stage-T RefinerNet trunk, AdapterS on the student's own bev_embed), trained with
       the stage-T GT surrogate on the refined trajectory tau_final (col 1, ttc 1, dac 1, prog 2, cmf 0.1, mod 0.1;
       m_col .15, m_dac .05, m_ttc .15, mode A, lon_st_slope .1).
  E2 : E1 + correction KD: L_KD = mean over the frozen teacher refiners (run-4 R_T on BEVFusion, R_M on ReSMap) of the
       mean |ctrl_S - ctrl_teacher| over 12 controls, same (detached, possibly perturbed) draft; weight
       lambda(t) = kd_lambda * clip((epoch_frac - a) / (b - a), 0, 1), (a, b) = kd_ramp.  The control space is
       cfg.kd_space and must be chosen explicitly for E2 training (no default):
         raw     : [z_lon, w_lat] (the plan; unbounded -> pulls the student's z_lon to the teachers' saturated +7..13,
                   where tanh' ~ 1e-7 also kills the surrogate's longitudinal gradient)
         tanh    : [tanh z_lon, tanh w_lat] (bounded, still pushes towards the positive dead zone, with a vanishing pull)
         decoded : [c_lon[2:] (m/s, after the mode-A clamp), e_lat[2:] (m)] of decode() (c_0 = c_1 = e_0 = e_1 = 0)
Gradient paths (report 36 M10): tau0 = sg(predictions['trajectory']) for student and teachers; bev_embed -> AdapterS
  through a x ref_bev_grad_scale (0.1) gradient scale; the surrogate / KD losses reach only R_S and (x0.1) the BEV
  encoder; PARA-SSR's own loss (incl. the grad balancer) is computed first and unchanged; the refiner losses are added
  after it.  Teachers: eval, frozen, fp32, no grad, not registered as modules (not in the checkpoint / optimiser).
Perturbed drafts: per micro-batch rng = np.random.default_rng([ref_seed, loss.iteration, rank]); per sample with
  probability ref_perturb_frac a family from the 12 non-identity slots of decoder.BANK_LAYOUT (sample_bank fallbacks),
  sample_perturbation(mode='A') on sg(tau0) whose path is extended by 8 straight poses at the end speed / heading
  (a student draft has no logged 8 s path); an invalid perturbation keeps tau0 (counted).
GT for the surrogate (target builder, per token; ref_gt_ok = all present): objects/{train,navtrain,dev}/<tok>.npz,
  sdf/navtrain/<tok>.npz, e2e_side/<tok[:2]>/<tok>.npz (cl_xy / cl_valid / cl_n = data.centerline_samples, p_pdm =
  PDM-Closed progress, tools/refiner/build_e2e_side.py).  The surrogate runs on the ok subset only (index-select).
Options added after the E2 pilot (defaults = the behaviour above, bit-identical; tools/refiner/tests/test_stageE.py)
  kd_balance 'ema' : KD weight at every micro-batch w = kd_ratio * EMA(L_sur_weighted) / max(EMA(L_KD), kd_ema_floor)
       (EMAs of the detached per-micro-batch values incl. GT-missing micro-batches (L_sur = 0), momentum kd_ema_m,
       bias-corrected, updated with the current micro-batch BEFORE the weight is taken; w is a python float = detached).
       Under DDP the two values are all-reduced (mean) before the update, so every rank holds the same EMA / weight.  A
       non-finite value skips the update.  KD is on from kd_start_epoch (None -> kd_ramp[0]) with no ramp; the EMAs run
       from step 0.  EMA state lives in the StageE callback state (Lightning checkpoint 'callbacks'), restored on resume.
  kd_draft_source 'human_mix' : the ref_perturb_frac samples that would get a perturbed sg(tau0) get the same stage-T
       bank perturbation of the GT human trajectory targets['trajectory'] instead (perturb_one: 12 non-identity
       families, sample_bank fallbacks, mode A, path = the logged 8 s human path / n_reg of HUMAN_NPZ exactly as
       make_draft_bank, delivered as ref_human_path / ref_human_nreg by GTLoader(human_path=True); a token not in
       HUMAN_NPZ -> the 8 poses + 8 straight poses, logged as ref/frac_human_logged_path); a non-finite / invalid one
       falls back to sg(tau0) (counted in ref/n_human_invalid).  Same draft for student and teachers, for surrogate and KD.
       Exact split with kd_draft_source human_mix: per sample an independent Bernoulli(ref_perturb_frac) draw (0.5 ->
       50 % in expectation, not an exact half of each micro-batch); selected -> perturbed GT human draft (invalid ->
       UNPERTURBED sg(tau0)); unselected -> UNPERTURBED sg(tau0) (human_mix never perturbs tau0).
  ref_human_only_until u (epoch, None = off) : while epoch_frac < u EVERY sample's draft (student and teachers, surrogate
       and KD, E1 and E2) is the stage-T bank perturbation of the GT human trajectory (same generator / logged path as
       human_mix, perturbation probability 1, ref_perturb_frac ignored); an invalid perturbation keeps the UNPERTURBED
       GT human trajectory (only a non-finite GT human trajectory falls back to sg(tau0)).  From epoch_frac >= u on the
       kd_draft_source behaviour above applies unchanged.  Logged per micro-batch: ref/human_only (1 / 0),
       ref/frac_draft_human, ref/frac_draft_tau0 (share of drafts built from the GT human trajectory (perturbed or
       not) / from sg(tau0) (perturbed or not)); ref/n_human, ref/n_human_invalid, ref/frac_human keep their human_mix
       meaning (perturbed human drafts / failed human perturbations).
  kd_ratio_ramp_epochs R (ema, None = off) : the ratio in the EMA weight is kd_ratio * clip((epoch_frac - kd_start) / R,
       0, 1) (per micro-batch), i.e. 0 at kd_start, kd_ratio from kd_start + R on.
  kd_weight_max C (ema, None = off) : w = min(ratio * EMA_hat(L_sur_w) / max(EMA_hat(L_KD), floor), C).  With either
       option set, kd/ratio and kd/w_ema_uncapped are logged too.
  grad_share_every n : every n micro-batches ||d L / d bev_embed|| of L_E0 (incl. the balancer correction), of the
       weighted surrogate (after the x0.1 scale) and of the weighted KD, via autograd.grad(retain_graph=True) (no .grad
       is touched); every micro-batch the value shares L / (L_E0 + L_sur_w + L_KD_w).
Inference (agent in eval mode): tau_final = decode(tau0, R_S(bev, tau0)) (no perturbation, mode A);
  predictions['trajectory'] = tau_final (ref_eval_traj='final', default) or tau0 ('tau0'); both are also returned as
  'tau_final' / 'tau0'.
"""
from __future__ import annotations

import importlib.util
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .adapters import BEV_H, BEV_W, HID_CH, OUT_CH
from .refiner_net import ADAPTER_SEED_OFFSET, RefinerNet

REPO = Path(__file__).resolve().parents[4]
TRAIN_REFINER = REPO / "tools/refiner/train_refiner.py"
MODES = ("off", "E1", "E2")
TERMS = ("col", "dac", "prog", "cmf", "mod", "ttc")
DEFAULT_TERM_WEIGHTS = {"col": 1.0, "ttc": 1.0, "dac": 1.0, "prog": 2.0, "cmf": 0.1, "mod": 0.1}
SIDE_DIR = "e2e_side"
KD_BEV_KEY = "kd_bev_{}"
KD_OK_KEY = "kd_ok_{}"
HUMAN_NPZ = "human/e2e_train_trainlogs.npz"   # extract_human.py extract --splits e2e_train_trainlogs (all 85,109)


# ----------------------------------------------------------------------------------------------- helpers
_TR = None


def train_refiner_module():
    """tools/refiner/train_refiner.py imported (not copied): scene_from_batch / surrogate_terms_batch / load_run_model."""
    global _TR
    if _TR is None:
        spec = importlib.util.spec_from_file_location("stageE_train_refiner", TRAIN_REFINER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _TR = mod
    return _TR


class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale: float):
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.scale, None


def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    return _GradScale.apply(x, scale) if x.requires_grad else x


class AdapterS(nn.Module):
    """Student BEV adapter: bev_embed [B, H*W, 256] (index = row * W + col = the stage-T S grid, row = x forward,
    col 0 = 32 m left; tests/test_stageE.py) -> gradient scale -> per-cell LayerNorm over 256 channels -> 1x1 256->128,
    GELU, 1x1 128->64 -> [B, 64, 50, 100].  Stateless normalisation (identical in train and eval)."""

    arm = "S"
    needs_bev = True

    def __init__(self, in_ch: int = 256, bev_grad_scale: float = 0.1, out_ch: int = OUT_CH):
        super().__init__()
        self.in_ch = in_ch
        self.bev_grad_scale = float(bev_grad_scale)
        self.ln = nn.LayerNorm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, HID_CH, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(HID_CH, out_ch, 1)
        self.out_ch = out_ch

    def forward(self, bev: torch.Tensor, n_tokens: Optional[int] = None) -> torch.Tensor:
        if bev is None or bev.dim() != 3 or bev.shape[1:] != (BEV_H * BEV_W, self.in_ch):
            raise ValueError(f"AdapterS needs bev_embed [B, {BEV_H * BEV_W}, {self.in_ch}], got "
                             f"{None if bev is None else tuple(bev.shape)}")
        B = bev.shape[0]
        x = self.ln(grad_scale(bev, self.bev_grad_scale).float())
        x = x.transpose(1, 2).reshape(B, self.in_ch, BEV_H, BEV_W)
        return self.conv2(self.act(self.conv1(x)))


def build_student(seed: int = 0, bev_grad_scale: float = 0.1, in_ch: int = 256) -> RefinerNet:
    """RefinerNet(arm='none', seed) with its adapter replaced by AdapterS; no global RNG is consumed."""
    with torch.random.fork_rng(devices=[]):
        net = RefinerNet("none", seed=seed)
        torch.manual_seed(seed + ADAPTER_SEED_OFFSET)
        net.adapter = AdapterS(in_ch, bev_grad_scale)
    net.arm = "S"
    return net


def ego_inputs(status_feature: torch.Tensor):
    """status_feature [B, 8] = command one-hot (left, straight, right, unknown), vx, vy, ax, ay (ego frame) ->
    v0 = |(vx, vy)|, a0 = ax, eds [B, 4], cmd index [B] (-1 if no command bit is set)."""
    sf = status_feature.float()
    oh, eds = sf[:, :4], sf[:, 4:8]
    cmd = torch.where(oh.sum(1) > 0, oh.argmax(1), torch.full_like(oh[:, 0], -1, dtype=torch.long))
    return torch.hypot(eds[:, 0], eds[:, 1]), eds[:, 2], eds, cmd.long()


# ----------------------------------------------------------------------------------------------- perturbation
def extend_straight(tau: np.ndarray, n: int = 8) -> np.ndarray:
    """[8, 3] draft -> [8 + n, 3] path: the draft, then n poses every 0.5 s straight on at the last segment speed and
    the final heading (stand-in for the logged 8 s path of stage T)."""
    tau = np.asarray(tau, np.float64)
    p7 = tau[-2, :2] if len(tau) > 1 else np.zeros(2)
    seg = float(np.hypot(*(tau[-1, :2] - p7)))
    h = float(tau[-1, 2])
    k = np.arange(1, n + 1)[:, None]
    ext = np.concatenate([tau[-1, :2] + k * seg * np.array([math.cos(h), math.sin(h)]), np.full((n, 1), h)], 1)
    return np.concatenate([tau, ext], 0)


def perturb_one(tau: np.ndarray, v0: float, a0: float, rng: np.random.Generator,
                path_long: Optional[np.ndarray] = None, n_valid: Optional[int] = None):
    """One stage-T bank perturbation of a draft (uniform over the 12 non-identity BANK_LAYOUT slots, sample_bank's
    fallbacks).  path_long / n_valid: the logged 8 s human path [16, 3] and its regular-grid length (human/<split>.npz
    'path' / 'n_reg', as make_draft_bank); None -> extend_straight(tau) (tau0 drafts have no logged path).
    -> (draft [8, 3] f32 or None if invalid, family name)."""
    from .decoder import BANK_LAYOUT, HumanContext, sample_perturbation

    fam = BANK_LAYOUT[1 + int(rng.integers(len(BANK_LAYOUT) - 1))]
    if path_long is None:
        ctx = HumanContext(tau, path_long=extend_straight(tau), v0=v0, a0=a0)
    else:
        ctx = HumanContext(tau, path_long, n_valid, v0, a0, None)
    f = fam
    if f == "ignore_brake" and not ctx.decel_ok:
        f = "lconst"
    if f == "creep" and not ctx.creep_ok:
        f = "lconst"
    if f in ("lat", "combined") and not ctx.lat_ok:
        f = "lconst"
    r = sample_perturbation(f, ctx, rng, mode="A")
    if not r["valid"] and f == "cv":
        r = sample_perturbation("hdrift", ctx, rng, mode="A")
    elif not r["valid"] and f in ("lat", "combined", "ignore_brake", "creep"):
        r = sample_perturbation("lconst", ctx, rng, mode="A")
    d = np.asarray(r["draft"], np.float32)
    if not r["valid"] or not np.isfinite(d).all():
        return None, fam
    return d, fam


def perturb_batch(tau0: torch.Tensor, v0: torch.Tensor, a0: torch.Tensor, frac: float, rng: np.random.Generator,
                  src: Optional[torch.Tensor] = None, src_path: Optional[torch.Tensor] = None,
                  src_nreg: Optional[torch.Tensor] = None):
    """-> (tau_in [B, 8, 3] (device / dtype of tau0), perturbed [B] bool, invalid [B] bool).  src (e.g. the GT human
    trajectory, kd_draft_source human_mix): perturb src instead of tau0; unselected / invalid samples keep tau0.
    src_path [B, 16, 3] / src_nreg [B] (with src): the logged human path per sample (make_draft_bank's path /
    n_reg); n_reg = -1 (token not in the human npz) -> extend_straight for that sample."""
    t = tau0.detach().double().cpu().numpy()
    s = t if src is None else src.detach().double().cpu().numpy()
    v, a = v0.detach().double().cpu().numpy(), a0.detach().double().cpu().numpy()
    out = t.astype(np.float32).copy()
    B = t.shape[0]
    P = None if src is None or src_path is None else src_path.detach().double().cpu().numpy()
    NR = None if P is None else src_nreg.detach().reshape(-1).long().cpu().numpy()
    pert, inval = np.zeros(B, bool), np.zeros(B, bool)
    for i in range(B):
        if rng.random() >= frac:
            continue
        try:
            if src is not None and not np.isfinite(s[i]).all():
                raise ValueError("non-finite source draft")
            if P is not None and int(NR[i]) >= 0:
                d, _ = perturb_one(s[i], float(v[i]), float(a[i]), rng, P[i], int(NR[i]))
            else:
                d, _ = perturb_one(s[i], float(v[i]), float(a[i]), rng)
        except Exception:          # degenerate draft (e.g. non-finite); counted as invalid
            d = None
        if d is None:
            inval[i] = True
        else:
            out[i], pert[i] = d, True
    return torch.as_tensor(out, device=tau0.device, dtype=tau0.dtype), torch.as_tensor(pert), torch.as_tensor(inval)


# ----------------------------------------------------------------------------------------------- GT loading
class GTLoader:
    """Per-token GT for the surrogate (and the teachers' BEVs for KD), padded to fixed shapes; used inside the target
    builder (dataloader workers).  Caches are opened lazily per process."""

    def __init__(self, data_root, teacher_runs: Sequence[str] = (), human_path: bool = False):
        from . import data as RD

        self.root = Path(data_root)
        # the three object stores are disjoint and together cover the E2E train_logs list (19,732 + 58,743 + 6,634 =
        # 85,109; the stage-T dev tokens exist only in objects/dev)
        self.src = SimpleNamespace(objects=[self.root / "objects" / d for d in ("train", "navtrain", "dev")],
                                   sdf=self.root / "sdf" / "navtrain")
        self.teacher_arms = [teacher_arm(r) for r in teacher_runs]
        self._caches = None
        self.human_path = bool(human_path)   # kd_draft_source human_mix: + ref_human_path / ref_human_nreg
        self._human = None
        self._A, self._CL, self._NKF = RD.A_MAX, RD.CL_MAX, RD.N_KF
        self._EH, self._EW = RD.E_H, RD.E_W

    def _teacher_caches(self):
        if self._caches is None:
            from . import data as RD
            from .resmap_cache import ResmapCache

            self._caches = [RD.TeacherCache.for_subset("navtrain") if a == "T" else ResmapCache.for_subset("navtrain")
                            for a in self.teacher_arms]
        return self._caches

    def _human_path(self, token: str):
        """-> (logged human path [16, 3] f32 (NaN beyond the log), n_reg) from HUMAN_NPZ; (zeros, -1) if absent."""
        if self._human is None:
            with np.load(self.root / HUMAN_NPZ, allow_pickle=False) as z:
                self._human = ({t: i for i, t in enumerate(z["tokens"].tolist())}, z["path"].astype(np.float32),
                               z["n_reg"].astype(np.int64))
        idx, path, nreg = self._human
        i = idx.get(token)
        if i is None:
            return np.zeros((16, 3), np.float32), np.int64(-1)
        return path[i], nreg[i]

    def _empty(self) -> Dict[str, np.ndarray]:
        A, CL, NKF = self._A, self._CL, self._NKF
        return {"obj_kf": np.zeros((A, NKF, 6), np.float32), "obj_first": np.zeros((A, 6), np.float32),
                "obj_meta": np.zeros((A, 5), np.int16), "obj_n": np.int32(0), "obj_n_kf": np.int16(NKF),
                "obj_R": np.float32(0), "obj_ego_kf": np.zeros((NKF, 3), np.float32),
                "sdf": np.zeros((self._EH, self._EW), np.float16), "cl_xy": np.zeros((CL, 2), np.float32),
                "cl_valid": np.zeros(CL, bool), "cl_n": np.int32(0), "p_pdm": np.float64(0.0)}

    def load(self, token: str) -> Dict[str, torch.Tensor]:
        from . import data as RD

        d, ok = self._empty(), True
        try:
            o = RD._read_objects(self.src, token)
            if o is None:
                ok = False
            else:
                d.update({k: o[k] for k in ("obj_kf", "obj_first", "obj_meta", "obj_n", "obj_n_kf", "obj_R",
                                            "obj_ego_kf")})
            s = RD._read_sdf(self.src, token)
            if s is None:
                ok = False
            else:
                d["sdf"] = s["sdf"]
            p = self.root / SIDE_DIR / token[:2] / f"{token}.npz"
            if p.is_file():
                with np.load(p, allow_pickle=False) as z:
                    d["cl_xy"], d["cl_valid"] = z["cl_xy"].astype(np.float32), z["cl_valid"].astype(bool)
                    d["cl_n"], d["p_pdm"] = np.int32(z["cl_n"]), np.float64(z["p_pdm"])
                ok = ok and bool(np.isfinite(d["p_pdm"])) and int(d["cl_n"]) >= 2
            else:
                ok = False
        except Exception:
            d, ok = self._empty(), False
        out = {f"ref_{k}": torch.as_tensor(np.asarray(v)) for k, v in d.items()}
        out["ref_gt_ok"] = torch.tensor(bool(ok))
        if self.human_path:              # a missing / unreadable HUMAN_NPZ raises (human_mix must not run silently without it)
            hp, hn = self._human_path(token)
            out["ref_human_path"], out["ref_human_nreg"] = torch.as_tensor(np.array(hp)), torch.tensor(int(hn))
        caches = self._teacher_caches() if self.teacher_arms else []   # a broken cache (root / sha / layout) raises
        for i, cache in enumerate(caches):
            try:
                bev, kok = np.asarray(cache.load_bev(token, s_grid=True), np.float16), True
            except Exception:
                bev, kok = np.zeros((256, BEV_H, BEV_W), np.float16), False
            out[KD_BEV_KEY.format(i)] = torch.as_tensor(np.ascontiguousarray(bev))
            out[KD_OK_KEY.format(i)] = torch.tensor(kok)
        return out


def teacher_arm(run_dir) -> str:
    arm = json.loads((Path(run_dir) / "config.json").read_text())["arm"]
    if arm not in ("T", "M"):
        raise ValueError(f"{run_dir}: KD teacher arm {arm!r} not in (T, M)")
    return arm


def surrogate_batch(targets: Dict[str, torch.Tensor], idx: torch.Tensor, tau_in: torch.Tensor, v0, a0) -> Dict:
    """ok-subset targets -> a data.collate_tokens-style batch (K = 1) for train_refiner.scene_from_batch; objects and
    centerline trimmed to the subset's maxima."""
    g = lambda k: targets[f"ref_{k}"][idx]
    n_obj = int(max(int(g("obj_n").max()), 1))
    n_cl = int(max(min(int(g("cl_n").max()), targets["ref_cl_xy"].shape[1]), 2))
    return {
        "tau0": tau_in[idx][:, None].float(), "human_traj": targets["trajectory"][idx].float(),
        "v0": v0[idx].float(), "a0": a0[idx].float(),
        "obj_kf": g("obj_kf")[:, :n_obj].float(), "obj_first": g("obj_first")[:, :n_obj].float(),
        "obj_meta": g("obj_meta")[:, :n_obj].long(),
        "obj_valid": torch.arange(n_obj, device=idx.device)[None] < g("obj_n").long()[:, None],
        "obj_n_kf": g("obj_n_kf").long(), "obj_R": g("obj_R").float(), "obj_ego_kf": g("obj_ego_kf").float(),
        "sdf": g("sdf"), "cl_xy": g("cl_xy")[:, :n_cl].float(), "cl_valid": g("cl_valid")[:, :n_cl].bool(),
        "pdm_progress_eff": g("p_pdm").double(),
    }


KD_SPACES = ("raw", "tanh", "decoded")
KD_BALANCES = ("fixed", "ema")
DRAFT_SOURCES = ("tau0", "human_mix")


def kd_controls(out: Dict, dec: Optional[Dict], space: str) -> torch.Tensor:
    """-> [B, 12] KD controls (first 6 longitudinal, last 6 lateral) in the chosen space (module docstring)."""
    if space == "raw":
        return torch.cat([out["z_lon"][:, 0], out["w_lat"][:, 0]], -1).float()
    if space == "tanh":
        return torch.tanh(torch.cat([out["z_lon"][:, 0], out["w_lat"][:, 0]], -1).float())
    if space == "decoded":
        return torch.cat([dec["c_lon"][:, 2:], dec["e_lat"][:, 2:]], -1).float()
    raise ValueError(f"kd_space {space!r} not in {KD_SPACES}")


def kd_loss(cs: torch.Tensor, cts: List[torch.Tensor], oks: List[torch.Tensor]):
    """cs [B, n] student controls, cts per-teacher [B, n] (detached here), oks per-teacher [B] bool ->
    (mean over teachers of mean |cs - ct| over its ok samples and the n controls, per-teacher list (0 without ok))."""
    cs = cs.float()
    per = []
    for ct, ok in zip(cts, oks):
        m = ok.to(cs.device).float().reshape(-1, 1)
        per.append(((cs - ct.float().detach()).abs() * m).sum() / torch.clamp(m.sum() * cs.shape[-1], min=1.0))
    return (sum(per) / len(per) if per else cs.new_zeros(())), per


# ----------------------------------------------------------------------------------------------- stage E
class StageE:
    """Loss / inference helper held by ParaSSRAgent (plain object: nothing here is a registered module)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = cfg.refiner_mode
        if self.mode not in MODES[1:]:
            raise ValueError(f"refiner_mode {self.mode!r} not in {MODES}")
        w = dict(DEFAULT_TERM_WEIGHTS)
        w.update(dict(cfg.ref_term_weights or {}))
        if set(w) - set(TERMS):
            raise ValueError(f"ref_term_weights keys {sorted(set(w) - set(TERMS))} not in {TERMS}")
        self.weights = w
        self.epoch_frac = 0.0
        self.rank = 0
        self._teachers: Optional[List] = None
        self._sur_cfg = None
        self.cum = {"gt_ok": 0, "gt_missing": 0, "perturbed": 0, "perturb_invalid": 0, "nonfinite": 0}
        self.skipped_steps = 0         # optimiser steps skipped for non-finite gradients (callback)
        self.kd_balance = getattr(cfg, "kd_balance", "fixed")
        self.draft_source = getattr(cfg, "kd_draft_source", "tau0")
        if self.kd_balance not in KD_BALANCES:
            raise ValueError(f"kd_balance {self.kd_balance!r} not in {KD_BALANCES}")
        if self.draft_source not in DRAFT_SOURCES:
            raise ValueError(f"kd_draft_source {self.draft_source!r} not in {DRAFT_SOURCES}")
        self.ema = {"sur": 0.0, "kd": 0.0, "n": 0}     # kd_balance 'ema' state (checkpointed by the callback)

    # -------------------------------------------------------------------------------------- pieces
    def sur_cfg(self):
        if self._sur_cfg is None:
            from .surrogate import SurrogateConfig
            c = self.cfg
            self._sur_cfg = SurrogateConfig(m_col=float(c.ref_m_col), m_dac=float(c.ref_m_dac), m_ttc=float(c.ref_m_ttc))
        return self._sur_cfg

    def teachers(self, device) -> List:
        if self._teachers is None:
            if not list(self.cfg.kd_teacher_runs or []):
                raise ValueError("refiner_mode E2 training needs kd_teacher_runs")
            if self.cfg.kd_space not in KD_SPACES:
                raise ValueError(f"refiner_mode E2 training needs an explicit kd_space in {KD_SPACES}, "
                                 f"got {self.cfg.kd_space!r}")
            tr = train_refiner_module()
            ts = []
            for run in self.cfg.kd_teacher_runs:
                net, _ = tr.load_run_model(Path(run), "best", "cpu")
                net.requires_grad_(False)
                ts.append(net.float().eval())
            self._teachers = ts
        for t in self._teachers:
            if next(t.parameters()).device != device:
                t.to(device)
        return self._teachers

    def kd_lambda(self) -> float:
        a, b = (float(x) for x in self.cfg.kd_ramp)
        r = 1.0 if b <= a else min(max((self.epoch_frac - a) / (b - a), 0.0), 1.0)
        if self.epoch_frac < a:
            r = 0.0
        return float(self.cfg.kd_lambda) * r

    # -------------------------------------------------------------------------------------- KD balance (ema)
    def kd_start(self) -> float:
        s = getattr(self.cfg, "kd_start_epoch", None)
        return float(self.cfg.kd_ramp[0]) if s is None else float(s)

    def ema_update(self, l_sur_w: float, l_kd: float, device=None) -> None:
        from ..modules.grad_balance import all_reduce_mean     # identity without an initialised process group
        v = all_reduce_mean({"sur": float(l_sur_w), "kd": float(l_kd)}, device or torch.device("cpu"))
        if not (math.isfinite(v["sur"]) and math.isfinite(v["kd"])):
            return
        m = float(self.cfg.kd_ema_m)
        self.ema["sur"] = m * self.ema["sur"] + (1.0 - m) * v["sur"]
        self.ema["kd"] = m * self.ema["kd"] + (1.0 - m) * v["kd"]
        self.ema["n"] += 1

    def ema_hat(self) -> Tuple[float, float]:
        n = int(self.ema["n"])
        if n == 0:
            return 0.0, 0.0
        c = 1.0 - float(self.cfg.kd_ema_m) ** n
        return self.ema["sur"] / c, self.ema["kd"] / c

    def kd_ratio_now(self) -> float:
        """kd_ratio, ramped linearly from 0 at kd_start over kd_ratio_ramp_epochs (if set) using epoch_frac."""
        r = float(self.cfg.kd_ratio)
        ramp = getattr(self.cfg, "kd_ratio_ramp_epochs", None)
        if ramp is None:
            return r
        return r * min(max((self.epoch_frac - self.kd_start()) / float(ramp), 0.0), 1.0)

    def ema_weight(self, capped: bool = True) -> float:
        """ratio * EMA_hat(L_sur_w) / max(EMA_hat(L_KD), floor) (0 before the first update), ratio = kd_ratio_now(),
        then min(., kd_weight_max) if set and capped."""
        if int(self.ema["n"]) == 0:
            return 0.0
        s, k = self.ema_hat()
        w = self.kd_ratio_now() * s / max(k, float(self.cfg.kd_ema_floor))
        cap = getattr(self.cfg, "kd_weight_max", None)
        if capped and cap is not None:
            w = min(w, float(cap))
        return w

    def ema_state(self) -> Dict:
        return {"ema_sur": float(self.ema["sur"]), "ema_kd": float(self.ema["kd"]), "ema_n": int(self.ema["n"])}

    def load_ema_state(self, sd: Dict) -> None:
        self.ema = {"sur": float(sd["ema_sur"]), "kd": float(sd["ema_kd"]), "n": int(sd["ema_n"])}

    @staticmethod
    def _run(net, bev, tau_in, ego):
        v0, a0, eds, cmd = ego
        return net(bev, tau_in[:, None], v0, a0, eds, cmd)

    def _decode(self, tau_in, out, v0, slope: float):
        from .decoder import decode
        return decode(tau_in.float(), out["z_lon"][:, 0].float(), out["w_lat"][:, 0].float(), v0=v0.float(),
                      mode="A", lon_st_slope=slope)

    # -------------------------------------------------------------------------------------- inference
    @torch.no_grad()
    def infer(self, student, features, predictions) -> Dict[str, torch.Tensor]:
        tau0 = predictions["trajectory"].detach()
        ego = ego_inputs(features["status_feature"])
        out = self._run(student, predictions["bev_embed"], tau0.float(), ego)
        tau_f = self._decode(tau0, out, ego[0], 0.0)["traj"].to(tau0.dtype)
        predictions = dict(predictions)
        predictions["tau0"], predictions["tau_final"] = tau0, tau_f
        predictions["trajectory"] = tau0 if self.cfg.ref_eval_traj == "tau0" else tau_f
        return predictions

    # -------------------------------------------------------------------------------------- training loss
    def loss(self, student, features, targets, predictions, iteration: int,
             log_bev_grad: bool = False, e0_loss: Optional[torch.Tensor] = None
             ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        t_start = time.time()
        bev = predictions["bev_embed"]
        dev = bev.device
        tau0 = predictions["trajectory"].detach().float()
        ego = ego_inputs(features["status_feature"])
        v0, a0 = ego[0], ego[1]
        B = tau0.shape[0]
        # perturbed drafts (sg(tau0) path; same draft for student and teachers)
        t_p = time.time()
        rng = np.random.default_rng([int(self.cfg.ref_seed), int(iteration), int(self.rank)])
        hou = getattr(self.cfg, "ref_human_only_until", None)
        human_only = hou is not None and self.epoch_frac < float(hou)
        human = self.draft_source == "human_mix" or human_only
        if human_only:      # every draft from the GT human trajectory (perturbed; invalid -> unperturbed GT human)
            src = targets["trajectory"].detach().float().to(tau0.device)
            hfin = torch.isfinite(src).flatten(1).all(1)
            base = torch.where(hfin[:, None, None], src, tau0)       # non-finite GT human only -> sg(tau0)
            tau_in, pert, inval = perturb_batch(base, v0, a0, 1.0, rng, src=src, src_path=targets.get("ref_human_path"),
                                                src_nreg=targets.get("ref_human_nreg"))
            h_draft = hfin.cpu()
        elif float(self.cfg.ref_perturb_frac) > 0:
            src = targets["trajectory"].detach().float().to(tau0.device) if human else None
            hp, hn = (targets.get("ref_human_path"), targets.get("ref_human_nreg")) if human else (None, None)
            tau_in, pert, inval = perturb_batch(tau0, v0, a0, float(self.cfg.ref_perturb_frac), rng, src=src,
                                                src_path=hp, src_nreg=hn)
        else:
            tau_in, pert, inval = tau0, torch.zeros(B, dtype=torch.bool), torch.zeros(B, dtype=torch.bool)
        t_pert = time.time() - t_p
        if hou is not None and not human_only:
            h_draft = pert.clone() if self.draft_source == "human_mix" else torch.zeros(B, dtype=torch.bool)
        out = self._run(student, bev, tau_in, ego)
        slope = float(self.cfg.ref_lon_st_slope)
        dec = self._decode(tau_in, out, v0, slope)
        logs: Dict[str, float] = {}
        # GT surrogate on the ok subset
        ok = targets["ref_gt_ok"].reshape(-1).bool().to(dev)
        idx = ok.nonzero()[:, 0]
        n_ok = int(idx.numel())
        l_sur = bev.new_zeros((), dtype=torch.float32)
        nonfinite = 0
        term_means = {k: 0.0 for k in TERMS + ("P1_minus_P0",)}
        if n_ok:
            tr = train_refiner_module()
            dec_s = dec if n_ok == B else self._decode(tau_in[idx], {k: out[k][idx] for k in ("z_lon", "w_lat")},
                                                       v0[idx], slope)
            sb = surrogate_batch(targets, idx, tau_in, v0, a0)
            terms = tr.surrogate_terms_batch(dec_s, sb, n_ok, 1, self.sur_cfg(), ttc_grad=bool(self.weights["ttc"]))
            # any non-finite raw term (before any nan_to_num) drops the whole micro-batch surrogate (counted); a
            # finite forward with a non-finite backward is caught by the callback (skipped optimiser step)
            bad = any(not bool(torch.isfinite(terms[k]).all()) for k in TERMS if self.weights[k])
            for k in TERMS:
                v = torch.nan_to_num(terms[k].float(), nan=0.0, posinf=1e3, neginf=-1e3).mean()
                term_means[k] = float(v.detach())
                if self.weights[k] and not bad:
                    l_sur = l_sur + self.weights[k] * v
            term_means["P1_minus_P0"] = float((terms["P1"] - terms["P0"]).float().mean().detach())
            if bad or not bool(torch.isfinite(l_sur)):
                l_sur, nonfinite = bev.new_zeros((), dtype=torch.float32), 1
        loss = float(self.cfg.ref_w) * l_sur
        sur_term, kd_term = loss, None
        # KD (E2)
        lam = 0.0
        if self.mode == "E2":
            lam = self.kd_lambda() if self.kd_balance == "fixed" else 0.0
            space = self.cfg.kd_space
            touts, cts, oks = [], [], []
            with torch.no_grad():
                for i, net in enumerate(self.teachers(dev)):
                    tb = targets[KD_BEV_KEY.format(i)].to(dev).float()
                    to = self._run(net, tb, tau_in, ego)
                    touts.append(to)
                    cts.append(kd_controls(to, self._decode(tau_in, to, v0, 0.0) if space == "decoded" else None,
                                           space))
                    oks.append(targets[KD_OK_KEY.format(i)].reshape(-1).bool())
            cs = kd_controls(out, dec, space)
            l_kd, per = kd_loss(cs, cts, oks)
            if self.kd_balance == "ema":
                self.ema_update(float(self.cfg.ref_w) * float(l_sur.detach()), float(l_kd.detach()), dev)
                lam = self.ema_weight() if self.epoch_frac >= self.kd_start() else 0.0
                s_hat, k_hat = self.ema_hat()
                logs["kd/ema_sur"], logs["kd/ema_kd"], logs["kd/w_ema"] = s_hat, k_hat, self.ema_weight()
                if (getattr(self.cfg, "kd_ratio_ramp_epochs", None) is not None
                        or getattr(self.cfg, "kd_weight_max", None) is not None):
                    logs["kd/ratio"], logs["kd/w_ema_uncapped"] = self.kd_ratio_now(), self.ema_weight(capped=False)
            kd_term = lam * l_kd
            loss = loss + kd_term
            logs["kd/loss"] = float(l_kd.detach())
            logs["kd/lambda"] = lam
            logs["kd/weighted"] = lam * float(l_kd.detach())
            from .decoder import lon_live
            for i, (p, to) in enumerate(zip(per, touts)):
                logs[f"kd/l1_{i}"] = float(p.detach())
                m = oks[i].to(dev).float()[:, None]
                den = float(torch.clamp(m.sum() * 6, min=1.0))
                d = (cs.detach() - cts[i]).abs()
                logs[f"kd/l1_lon_{i}"] = float((d[:, :6] * m).sum()) / den
                logs[f"kd/l1_lat_{i}"] = float((d[:, 6:] * m).sum()) / den
                logs[f"kd/teacher_live_{i}"] = float(lon_live(to["z_lon"][:, 0].float()).float().mean())
                logs[f"kd/n_ok_{i}"] = float(oks[i].sum())
        # every student parameter joins the graph (unused gate head; E1 micro-batches without GT) -> DDP-safe
        loss = loss + 0.0 * (out["gate_logit"].sum() + out["z_lon"].sum() + out["w_lat"].sum())
        # diagnostics
        with torch.no_grad():
            live = (dec["c_lon"] < 0).any(-1).float()
            arc = lambda tr_: torch.cat([torch.zeros_like(tr_[:, :1, :2]), tr_[:, :, :2]], 1).diff(dim=1).norm(dim=-1).sum(1)
            logs.update({
                "ref/live": float(live.mean()), "ref/short_m": float((arc(tau_in) - arc(dec["traj"])).mean()),
                "ref/lat_m": float(dec["d"].abs().amax(1).mean()),
                "ref/dist_final_tau0": float((dec["traj"][..., :2] - tau_in[..., :2]).norm(dim=-1).mean()),
                "ref/zdead": float((torch.relu(out["z_lon"].float()) ** 2).mean()),
            })
        for k in TERMS:
            logs[f"ref/t_{k}"] = term_means[k]
        logs["ref/P1_minus_P0"] = term_means["P1_minus_P0"]
        logs["ref/L_sur"] = float(l_sur.detach())
        logs["ref/L_sur_weighted"] = float(self.cfg.ref_w) * float(l_sur.detach())
        counts = {"gt_ok": n_ok, "gt_missing": B - n_ok, "perturbed": int(pert.sum()),
                  "perturb_invalid": int(inval.sum()), "nonfinite": nonfinite}
        for k, v in counts.items():
            self.cum[k] += v
            logs[f"ref/n_{k}"] = float(v)
            logs[f"ref/cum_{k}"] = float(self.cum[k])
        if human:
            logs["ref/n_human"] = float(pert.sum())
            logs["ref/n_human_invalid"] = float(inval.sum())
            logs["ref/frac_human"] = float(pert.sum()) / max(B, 1)
            hn = targets.get("ref_human_nreg")
            logs["ref/frac_human_logged_path"] = (float((hn.reshape(-1) >= 0).float().mean()) if hn is not None
                                                  else 0.0)
        if hou is not None:
            fh = float(h_draft.float().sum()) / max(B, 1)
            logs["ref/human_only"] = float(human_only)
            logs["ref/frac_draft_human"], logs["ref/frac_draft_tau0"] = fh, 1.0 - fh
        if e0_loss is not None:
            vals = {"e0": float(e0_loss.detach()), "sur": float(sur_term.detach()),
                    "kd": 0.0 if kd_term is None else float(kd_term.detach())}
            tot = sum(abs(x) for x in vals.values()) or 1.0
            for k, x in vals.items():
                logs[f"ref/vshare_{k}"] = x / tot
            every = int(getattr(self.cfg, "grad_share_every", 0) or 0)
            if every > 0 and iteration % every == 0 and bev.requires_grad and torch.is_grad_enabled():
                gn = {}
                for k, t in (("e0", e0_loss), ("sur", sur_term), ("kd", kd_term)):
                    g = None
                    if isinstance(t, torch.Tensor) and t.requires_grad:
                        g = torch.autograd.grad(t, bev, retain_graph=True, allow_unused=True)[0]
                    gn[k] = 0.0 if g is None else float(g.float().norm())
                gtot = sum(gn.values()) or 1.0
                for k, x in gn.items():
                    logs[f"gnorm/bev_{k}"] = x
                    logs[f"ref/gshare_{k}"] = x / gtot
        if log_bev_grad and bev.requires_grad:
            g = torch.autograd.grad(loss, bev, retain_graph=True, allow_unused=True)[0]
            logs["gnorm/ref_bev"] = 0.0 if g is None else float(g.float().norm())
        logs["time/ref_perturb_ms"] = 1e3 * t_pert
        logs["time/ref_ms"] = 1e3 * (time.time() - t_start)
        logs["ref/loss"] = float(loss.detach())
        return loss, {k: torch.tensor(float(v), device=dev) for k, v in logs.items()}


# ----------------------------------------------------------------------------------------------- lightning hooks
def refiner_parameters(agent) -> List[nn.Parameter]:
    return [p for p in agent.ref_student.parameters() if p.requires_grad]


def make_callback(agent):
    import pytorch_lightning as pl

    class StageECallback(pl.Callback):
        """Fractional epoch for lambda_KD, rank for the perturbation rng, the refiner-only gradient clip that runs
        before Lightning's global clip (on_before_optimizer_step precedes _clip_gradients in PL 2.x), and (rank 0) a
        per-micro-batch record <root>/stageE_steps.jsonl (refiner / KD logs, loss, step and data-wait seconds, peak
        GPU memory) for the pilot rule (tools/refiner/stageE_prep.py)."""

        def __init__(self):
            super().__init__()
            self._f = None
            self._t_start = self._t_end = None
            self._n = 0

        def state_dict(self):
            # kd_balance 'ema' only (empty otherwise -> Lightning stores nothing, as before)
            st = agent._stage_e
            return st.ema_state() if st.kd_balance == "ema" else {}

        def load_state_dict(self, state_dict):
            if state_dict and agent._stage_e.kd_balance == "ema":
                agent._stage_e.load_ema_state(state_dict)

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            n = max(1, int(trainer.num_training_batches or 1))
            agent._stage_e.epoch_frac = float(trainer.current_epoch) + float(batch_idx) / n
            agent._stage_e.rank = int(getattr(trainer, "global_rank", 0))
            self._t_start = time.time()

        def _log_scalars(self, pl_module, now):
            """The jsonl-only fields as stageE/* scalars (TensorBoard + W&B).  The refiner / KD / time/ref_* values of
            latest_logs already reach the loggers as train/ref/*, train/kd/*, train/gnorm/*, train/time/ref_*
            (ParaSSRLoggingCallback).  Called on every rank (same keys), rank_zero_only + no sync_dist: no
            collective; batch_size=1: epoch means are per micro-batch and nothing is inferred from the batch."""
            st = agent._stage_e
            vals = {"stageE/epoch_frac": (st.epoch_frac, "mean", False),
                    "stageE/skipped_steps": (st.skipped_steps, "max", True)}
            if self._t_start:
                vals["stageE/sec_step"] = (now - self._t_start, "mean", True)
                if self._t_end:
                    vals["stageE/sec_wait"] = (self._t_start - self._t_end, "mean", True)
            if torch.cuda.is_available():
                vals["stageE/mem_gb"] = (torch.cuda.max_memory_allocated() / 2 ** 30, "max", True)
            for k, (v, fx, ep) in vals.items():
                pl_module.log(k, float(v), on_step=True, on_epoch=ep, reduce_fx=fx, prog_bar=False,
                              sync_dist=False, rank_zero_only=True, batch_size=1)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            now = time.time()
            self._log_scalars(pl_module, now)
            if getattr(trainer, "is_global_zero", True):
                if self._f is None:
                    root = getattr(getattr(trainer, "logger", None), "save_dir", None) or trainer.default_root_dir
                    Path(root).mkdir(parents=True, exist_ok=True)
                    self._f = open(Path(root) / "stageE_steps.jsonl", "a")
                rec = {"epoch": int(trainer.current_epoch), "batch": int(batch_idx), "gstep": int(trainer.global_step),
                       "epoch_frac": agent._stage_e.epoch_frac,
                       "sec_step": round(now - self._t_start, 4) if self._t_start else None,
                       "sec_wait": round(self._t_start - self._t_end, 4) if (self._t_end and self._t_start) else None,
                       "skipped_steps": agent._stage_e.skipped_steps}
                if torch.cuda.is_available():
                    rec["mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
                for k, v in (agent.latest_logs or {}).items():
                    if k.startswith(("ref/", "kd/", "time/", "gnorm/", "loss")):
                        rec[k] = float(v)
                self._f.write(json.dumps(rec) + "\n")
                self._n += 1
                if self._n % 50 == 0:
                    self._f.flush()
            self._t_end = now

        def on_train_end(self, trainer, pl_module):
            if self._f is not None:
                self._f.close()
                self._f = None

        def on_exception(self, trainer, pl_module, exception):
            self.on_train_end(trainer, pl_module)

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            # non-finite gradient anywhere (grads are already all-reduced under DDP, so every rank agrees): skip this
            # optimiser step (grad = None -> AdamW leaves the parameter and its state untouched) and log it
            allp = [p for p in pl_module.parameters() if p.grad is not None]
            if allp:
                gn_all = torch.stack([g.float() for g in torch._foreach_norm([p.grad for p in allp])])
                if not bool(torch.isfinite(gn_all).all()):
                    for p in allp:
                        p.grad = None
                    agent._stage_e.skipped_steps += 1
                    pl_module.log("train/ref/skipped_nonfinite", float(agent._stage_e.skipped_steps), on_step=True,
                                  rank_zero_only=True)
                    if getattr(trainer, "is_global_zero", True):
                        print(f"[stageE] non-finite gradient at global_step {trainer.global_step}: optimiser step "
                              f"skipped (total {agent._stage_e.skipped_steps})", flush=True)
                    return
            clip = float(agent.config.ref_clip)
            params = [p for p in refiner_parameters(agent) if p.grad is not None]
            if clip > 0 and params:
                gn = torch.nn.utils.clip_grad_norm_(params, clip)
                pl_module.log("train/ref/grad_norm_preclip", float(gn), on_step=True, on_epoch=True,
                              rank_zero_only=True)

    return StageECallback()
