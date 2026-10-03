#!/usr/bin/env python
"""Train one stage-T refiner (IMPL_SPEC §3.8): arm {T, none, M, TM} x fold x seed.

  python tools/refiner/train_refiner.py --arm T --fold -1 --seed 0 --gpu 3 [--workers 2]

Data     : packed split (navsim/agents/para_ssr/refiner/data.py; <data>/packed/train), token-batched 8 tokens x 13
           drafts; arm T additionally reads the teacher bev_feature per token (manifest sha head cddf943ffec8d6a8).
Rows     : train split, drafts + labels + human + objects + sdf + centerline packed, no frame gap.  --fold k (0..4)
           trains on folds != k (cross-fitting; fold k is the out-of-fold set for eval_refiner.py), --fold -1 on all
           folds.  Early stopping on an inner-validation set of whole TRAINING logs (data.inner_val_logs, 10 %,
           hash-based, the same for every arm and seed).
Norm     : arm T per-channel z-score over <= --n-norm inner-train tokens (seeded), saved to <run>/norm.npz (reused
           on resume).
Model    : RefinerNet (refiner_net.py); forward under fp16 autocast on GPU; decoder (decoder.decode, mode A by default)
           and losses in float32 (compute_loss runs outside the autocast region; decode_batch casts to float32).
Loss     : sum_n w_n * mean_valid(term_n) over valid drafts (draft_valid) + w_gate * BCE(gate_logit, y),
           y = 1[nc < 1 or dac < 1 or ddc < 1] (+ ttc < 1 with --gate-ttc), pos_weight = min((1 - pi) / pi, 10) with pi
           the label prevalence over the run's training drafts.  The gate head reads the detached trunk (refiner_net).
           Correction terms (per draft [B]) come from the surrogate (IMPL_SPEC §3.7): col, dac, prog, cmf, mod.
           Optional (run 2, default off): + w_zdead * zdead, and the mode-A straight-through slope (--lon-st-slope).
Optimiser: AdamW lr 3e-4, wd 0.01, cosine decay over epochs x steps (per-step), epochs 40, early stop patience
           --patience epochs on the inner-val total loss, best checkpoint kept.  Gradient-norm clip --clip (1.0).

SURROGATE (resolved at start, recorded in <run>/config.json as 'surrogate'):
  --surrogate real (DEFAULT since the integration step, 2026-09-28; a missing / broken surrogate.py is an error, never
  a silent fallback) | auto: navsim.agents.para_ssr.refiner.surrogate.surrogate_terms(dec, tau0, scene, index, cfg)
    with scene = scene_from_batch(batch) (surrogate.SceneBatch built on the device from the packed batch: objects via
    gt_future.query_torch at t = 0.1 n, OBS mask, is_agent, human_overlap = surrogate.human_overlap_mask of the human
    dense reference, gt_ego / t_avail / R for the UNKNOWN test, E-grid SDF, the surrogate's own centerline crop,
    p_pdm = pdm_progress_eff, v0, a0) and index = token of each draft.  Equality with the surrogate's reference
    builder (scene_from_numpy + collate_scenes) is tested in tests/test_train_refiner.py.  Terms per draft [B]:
    col, dac, prog, cmf, mod (+ P1, P0, unknown logged with weight 0).
  --m-col / --m-dac (optional; unset = SurrogateConfig() code defaults 0.3 / 0.2, unchanged): surrogate margin
    overrides (M8 re-check, report/refiner_T/m8_recheck/), used in training AND inner-val losses; recorded in
    config.json ('m_col', 'm_dac', 'surrogate_cfg'); resuming a run with different values is refused; stub: error.
  --lon-st-slope lam / --w-zdead w_zd (PRESTATED_DECISION_RULE AMENDMENT 3, run 2; defaults 0 = run 1): mode-A
    dead-zone fix for TRAINING losses only.  lam: decoder.decode(..., lon_st_slope=lam) passes a straight-through gradient
    of slope lam through the clamp c <- min(c, 0) (the forward, hence every trajectory and loss value, is unchanged;
    eval_refiner decodes without it); w_zd adds w_zd * zdead, zdead = mean over valid drafts of mean_i relu(z_lon_i)^2
    (raw lon head output, 6 controls).  Both in config.json; resuming with other values is refused.  Logged per step /
    epoch: 'zdead' (unweighted), 'live' (fraction of valid drafts with any decoded c_lon < 0 = decoder.lon_live);
    epoch records also carry val_liveness / train_liveness (valid-draft weighted) and val 'loss_ex_zd' (= loss minus
    w_zd * zdead, comparable across w_zd; 'loss', used for early stopping, is the full objective).
  --w ...,ttc=<w> / --m-ttc m (PRESTATED_DECISION_RULE AMENDMENT 4, run 3; defaults w ttc = 0, m_ttc unset =
    SurrogateConfig().m_ttc = 0 -> runs 1 / 2 unchanged): TTC surrogate term surrogate.ttc_cost (projected ego boxes at
    0.3 / 0.6 / 0.9 s vs the objects at n + 3 / 6 / 9; scene_from_batch adds the 0..5 s objects boxes_ttc / obs_ttc and
    the human dense reference).  The term is always computed and logged as 't_ttc' (step / epoch; unweighted), and
    enters the loss only with a non-zero weight.  weights['ttc'] and m_ttc are in config.json; resuming with other
    values is refused (a config.json without them = 0 / unset).  The stub has no TTC term (ttc = 0): a non-zero TTC
    weight or --m-ttc with --surrogate stub is refused.
  --surrogate stub: stub_terms below (plumbing only; 'mod' and 'dac' follow §3.7, 'col' is a disc approximation,
    'prog' an arc-length shortfall, 'cmf' analytic only).  'auto' falls back to the stub only if surrogate.py is
    missing (opt-in only).  Default weights = surrogate.DEFAULT_WEIGHTS (col 1, dac 1, prog 2, cmf 0.1, mod 0.1; gate 0.5) --
    placeholders to be chosen by the 5-fold cross-fitting (PRESTATED_DECISION_RULE.txt).
RUN 4 (PRESTATED_DECISION_RULE AMENDMENT 6; every option below is off by default -> runs 1-3 unchanged):
  --arm M   ReSMap map-teacher neck BEV (resmap_cache.ResmapCache, S grid), AdapterM; z-score -> <run>/norm_map.npz.
  --arm TM  BEVFusion + ReSMap (data.ConcatTeacher, [512, 50, 100]), AdapterTM; norm.npz (det) + norm_map.npz (map).
            Both norms are computed exactly like arm T's (data.compute_teacher_norm over <= --n-norm of the run's
            inner-train tokens, seed 0), so a TM run's norm.npz equals the T run's of the same subset / fold; each file's
            meta records its cache root and sha head.
  --token-subset <parquet>  restrict the packed rows to the parquet's tokens BEFORE the inner-val selection (the fold
            filter and the token filter commute; inner_val_logs is then computed on the restricted logs, hash-based,
            deterministic).  config.json 'token_subset' = {path, sha256, n_rows, n_tokens, n_packed_rows (trainable rows
            kept, all folds), n_train_rows, n_ival_rows}; resuming with another subset (sha256) is refused.
            Arms M / TM REQUIRE it and every token of it must be in the ReSMap cache (navtrain root = train_logs only).
  --dev-token-subset <parquet>  recorded only (config.json 'dev_token_subset' = {path, sha256, n_rows, n_tokens}): the
            default dev subset of eval_refiner.py --split dev for this run.
  --resmap-root <dir>  override the ReSMap cache directory (tests; meta.json sha256-checked).
Outputs  : <runs>/<run>/{config.json, norm.npz (arms T, TM), norm_map.npz (arms M, TM), log.jsonl, ckpt_last.pt,
           ckpt_best.pt, summary.json, DONE}.
           Resumable: rerunning the same command continues from ckpt_last.pt (epoch granularity); a finished run
           (DONE) exits immediately (code 0) unless --restart -- but only if its config.json matches this call's
           m_col / m_dac / lon_st_slope / w_zdead / m_ttc / weights['ttc']; a mismatch is refused (non-zero exit) even
           for a finished run.

Deviations (documented): gradient clipping (--clip 1.0) and a final-LayerNorm in the net are additions not in the
spec; the loss weights are placeholders; --warmup-frac defaults to 0 (pure cosine as in the spec); the gate BCE is
averaged over valid drafts only (draft_valid), with pi from the run's training drafts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from navsim.agents.para_ssr.refiner import data as RD  # noqa: E402
from navsim.agents.para_ssr.refiner.adapters import load_norm, save_norm  # noqa: E402
from navsim.agents.para_ssr.refiner.decoder import decode  # noqa: E402
from navsim.agents.para_ssr.refiner.geometry import HALF_LEN, HALF_WID, RA2C, dense_reference  # noqa: E402
from navsim.agents.para_ssr.refiner.gt_future import KF_DT, T_MAX, query_torch  # noqa: E402
from navsim.agents.para_ssr.refiner.refiner_net import RefinerNet  # noqa: E402
from navsim.agents.para_ssr.refiner.sdf import corner_sdf  # noqa: E402

TERMS = ("col", "dac", "prog", "cmf", "mod", "ttc")
DEFAULT_W = {"col": 1.0, "dac": 1.0, "prog": 2.0, "cmf": 0.1, "mod": 0.1,      # = surrogate.DEFAULT_WEIGHTS (placeholders)
             "ttc": 0.0}                                                        # AMENDMENT 4: off unless --w ttc=...
DEFAULT_W_GATE = 0.5
RUNS = RD.DATA_ROOT / "runs"
POS_WEIGHT_CAP = 10.0
CODE_FILES = ("adapters.py", "corridor.py", "refiner_net.py", "data.py", "decoder.py", "geometry.py", "sdf.py",
              "gt_future.py", "surrogate.py")

# stub constants (IMPL_SPEC §3.7 values where the stub follows the spec)
M_DAC, M_COL, BETA = 0.2, 0.3, 0.1
NOT_BEHIND = -1.3
COMF_JERK, COMF_DA = 4.13, 4.05


# ----------------------------------------------------------------------------------------------- surrogate
def stub_terms(dec: Dict, batch: Dict, n_tokens: int, n_drafts: int, cfg: Optional[Dict] = None,
               ttc_grad: bool = True) -> Dict[str, torch.Tensor]:
    """STUB correction terms per draft [B] (plumbing only, see module docstring; the real terms are surrogate_terms_batch)."""
    T, K = n_tokens, n_drafts
    dense = dec["dense"]                                                     # [B, 41, 3]
    B, NT = dense.shape[:2]
    dev, dt = dense.device, dense.dtype
    tidx = torch.arange(T, device=dev).repeat_interleave(K)
    # modification (spec): mean_n |s1 - s0| / 5 + mean_n |d| / 1
    mod = (dec["s"] - dec["s0"]).abs().mean(1) / 5.0 + dec["d"].abs().mean(1)
    # DAC (spec form): corners of the 41 dense poses, bilinear E-grid SDF, out-of-grid corners excluded
    val, ok = corner_sdf(batch["sdf"], dense, sdf_index=tidx)                # [B, 41, 4]
    dac = (BETA * F.softplus((M_DAC - val.to(dt)) / BETA) * ok.to(dt)).sum(-1).mean(-1)
    # collision (STUB: discs of radius half-width around the box centres; OBS only; not-behind; human-overlap mask)
    t = torch.arange(NT, device=dev, dtype=torch.float64) * 0.1
    boxes, state = query_torch(batch["obj_kf"], batch["obj_first"], batch["obj_meta"], t,
                               valid=batch["obj_valid"], n_kf=batch["obj_n_kf"])  # [T, A, 41, 5], [T, A, 41]
    boxes = boxes.to(dt)
    agent = batch["obj_meta"][..., 1].to(dt)                                 # [T, A]
    w_obj = torch.where(agent > 0, torch.ones_like(agent), 0.5 * torch.ones_like(agent))

    def centre(poses):
        return poses[..., :2] + RA2C * torch.stack([torch.cos(poses[..., 2]), torch.sin(poses[..., 2])], -1)

    def gap(c_ego, bx):                                                       # c_ego [T|B, 41, 2], bx [T|B, A, 41, 5]
        dist = torch.linalg.norm(c_ego[:, None] - bx[..., :2] + 1e-9, dim=-1)
        return dist - HALF_WID - 0.5 * torch.minimum(bx[..., 3], bx[..., 4])

    ce = centre(dense)                                                        # [B, 41, 2]
    bx = boxes[tidx]
    g = gap(ce, bx)                                                           # [B, A, 41]
    hd = dense[..., 2]
    along = ((bx[..., 0] - ce[:, None, :, 0]) * torch.cos(hd)[:, None] + (bx[..., 1] - ce[:, None, :, 1]) * torch.sin(hd)[:, None])
    g_h = gap(centre(dense_reference(batch["human_traj"].to(dt))), boxes)[tidx]
    mask = (state[tidx] > 0) & (along >= NOT_BEHIND) & (g_h >= 0)
    col = (w_obj[tidx][..., None] * BETA * F.softplus((M_COL - g) / BETA) * mask.to(dt)).sum(1).mean(-1)
    # progress (STUB: shortfall of the decoded arc length at 4 s vs the draft)
    s0e = dec["s0"][:, -1]
    prog = torch.relu(s0e - dec["s"][:, -1]) / torch.clamp(s0e, min=5.0)
    # comfort (STUB: analytic delta-accel and jerk of the correction only)
    comf = ((torch.relu(dec["jerk"].abs() - 0.9 * COMF_JERK) / COMF_JERK) ** 2).mean(1) + \
        ((torch.relu(dec["da"].abs() - 0.9 * COMF_DA) / COMF_DA) ** 2).mean(1)
    return {"col": col, "dac": dac, "prog": prog, "cmf": comf, "mod": mod, "ttc": torch.zeros_like(mod)}   # no TTC stub


def gt_ego_torch(ego_kf: torch.Tensor, n_kf: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """torch twin of gt_future.gt_ego_xy: ego_kf [S, 11, 3], n_kf [S], t [T] -> [S, T, 2] (np.interp, clamped)."""
    S = ego_kf.shape[0]
    n = n_kf.long().clamp(min=1).to(ego_kf.device)
    u = torch.minimum((t.to(ego_kf.dtype) / KF_DT)[None].expand(S, -1), (n - 1).to(ego_kf.dtype)[:, None]).clamp(min=0)
    k0 = torch.minimum(torch.floor(u).long(), (n - 2).clamp(min=0)[:, None])
    k1 = torch.minimum(k0 + 1, (n - 1)[:, None])
    w = (u - k0.to(u.dtype)).clamp(0, 1)[..., None]
    g = lambda k: torch.gather(ego_kf[..., :2], 1, k[..., None].expand(-1, -1, 2))
    return g(k0) * (1 - w) + g(k1) * w


def scene_from_batch(batch: Dict, dtype=torch.float32):
    """data.collate_tokens batch (on its device) -> surrogate.SceneBatch (one scene per token)."""
    from navsim.agents.para_ssr.refiner import surrogate as S

    dev = batch["tau0"].device
    t = torch.arange(41, device=dev, dtype=torch.float64) * 0.1
    # boxes and the (exact, touching counts) human-overlap mask in float64 like surrogate.scene_from_numpy, then cast
    boxes, state = query_torch(batch["obj_kf"].double(), batch["obj_first"].double(), batch["obj_meta"], t,
                               valid=batch["obj_valid"], n_kf=batch["obj_n_kf"])
    hd = dense_reference(batch["human_traj"].double())
    hov = S.human_overlap_mask(hd, boxes) & batch["obj_valid"][..., None]
    # TTC objects at t = 0.1 m, m = 0..50 (0..5 s); a separate query so that boxes / obs above stay exactly as before
    t_ttc = torch.arange(S.NK_OBJ, device=dev, dtype=torch.float64) * 0.1
    boxes_t, state_t = query_torch(batch["obj_kf"].double(), batch["obj_first"].double(), batch["obj_meta"], t_ttc,
                                   valid=batch["obj_valid"], n_kf=batch["obj_n_kf"])
    n_kf = batch["obj_n_kf"]
    return S.SceneBatch(
        boxes=boxes.to(dtype), obs=state == 1, is_agent=(batch["obj_meta"][..., 1] > 0) & batch["obj_valid"],
        human_overlap=hov, sdf=batch["sdf"], centerline=batch["cl_xy"].to(dtype),
        cl_valid=batch["cl_valid"], p_pdm=batch["pdm_progress_eff"].to(dtype), v0=batch["v0"].to(dtype),
        a0=batch["a0"].to(dtype), gt_ego=gt_ego_torch(batch["obj_ego_kf"].to(dtype), n_kf, t),
        t_avail=((n_kf - 1).to(dtype) * KF_DT).clamp(max=T_MAX), R=batch["obj_R"].to(dtype),
        boxes_ttc=boxes_t.to(dtype), obs_ttc=state_t == 1, human_dense=hd)


def surrogate_terms_batch(dec: Dict, batch: Dict, n_tokens: int, n_drafts: int, cfg=None,
                          ttc_grad: bool = True) -> Dict[str, torch.Tensor]:
    """Real M7 terms for the T*K drafts of a batch (surrogate.surrogate_terms on scene_from_batch).  ttc_grad=False:
    the TTC term is computed without autograd (compute_loss passes it when the TTC weight is 0; values unchanged)."""
    from navsim.agents.para_ssr.refiner import surrogate as S

    dt = dec["dense"].dtype
    scene = scene_from_batch(batch, dt)
    tidx = torch.arange(n_tokens, device=dec["dense"].device).repeat_interleave(n_drafts)
    tau0 = batch["tau0"].reshape(n_tokens * n_drafts, 8, 3).to(dt)
    out = S.surrogate_terms(dec, tau0, scene, index=tidx, cfg=cfg if cfg is not None else S.SurrogateConfig(),
                            ttc_grad=ttc_grad)
    return {k: v for k, v in out.items() if k != "details"}


def resolve_surrogate(name: str = "auto") -> Tuple[Callable, str]:
    """-> (terms_fn, label).  'real' requires surrogate.surrogate_terms; 'auto' falls back to the stub only if the
    surrogate module is missing."""
    if name in ("auto", "real"):
        try:
            from navsim.agents.para_ssr.refiner import surrogate as S
            getattr(S, "surrogate_terms"), getattr(S, "SceneBatch"), getattr(S, "human_overlap_mask")
            return surrogate_terms_batch, "surrogate.surrogate_terms"
        except (ImportError, AttributeError) as e:
            if name == "real":
                raise RuntimeError(f"surrogate.surrogate_terms not available: {type(e).__name__}: {e}") from e
    return stub_terms, "stub"


def surrogate_config(a):
    """Optional surrogate margin overrides (--m-col / --m-dac, added 2026-09-28 for the M8 re-check, STATUS.md D1;
    --m-ttc, AMENDMENT 4).  None when none is given -> surrogate_terms_batch uses SurrogateConfig() (code defaults,
    unchanged).  Otherwise SurrogateConfig with ONLY the given fields replaced; every other field keeps its default."""
    over = {k: v for k, v in (("m_col", getattr(a, "m_col", None)), ("m_dac", getattr(a, "m_dac", None)),
                              ("m_ttc", getattr(a, "m_ttc", None))) if v is not None}
    if not over:
        return None
    from navsim.agents.para_ssr.refiner import surrogate as S
    return S.SurrogateConfig(**over)


def surrogate_config_record(cfg, sur_label: str):
    """effective surrogate config for config.json (None for the stub, which ignores it)."""
    if sur_label == "stub":
        return None
    import dataclasses
    from navsim.agents.para_ssr.refiner import surrogate as S
    return dataclasses.asdict(cfg if cfg is not None else S.SurrogateConfig())


# ----------------------------------------------------------------------------------------------- losses
def gate_targets(labels: torch.Tensor, use_ttc: bool = False) -> torch.Tensor:
    """y [T, K] = 1[nc < 1 or dac < 1 or ddc < 1 (or ttc < 1)] from labels [T, K, L] (data.LABEL_COLS)."""
    L = RD.LBL
    fail = (labels[..., L["nc"]] < 1) | (labels[..., L["dac"]] < 1) | (labels[..., L["ddc"]] < 1)
    if use_ttc:
        fail = fail | (labels[..., L["ttc"]] < 1)
    return fail.float()


def label_prevalence(packed: RD.PackedSplit, rows, use_ttc: bool = False) -> float:
    lab = torch.as_tensor(np.asarray(packed.arrays["labels"][np.sort(rows)]))
    val = torch.as_tensor(np.asarray(packed.arrays["draft_valid"][np.sort(rows)]))
    y = gate_targets(lab, use_ttc)
    return float(y[val].mean()) if bool(val.any()) else float("nan")


def decode_batch(out: Dict, batch: Dict, mode: str = "A", lon_st_slope: float = 0.0) -> Dict:
    """float32 decode of the T*K drafts of a batch.  lon_st_slope: mode-A straight-through backward slope (training
    losses only; the forward does not depend on it)."""
    T, K = batch["tau0"].shape[:2]
    return decode(batch["tau0"].reshape(T * K, 8, 3).float(), out["z_lon"].float().reshape(T * K, -1),
                  out["w_lat"].float().reshape(T * K, -1), v0=batch["v0"].float().repeat_interleave(K), mode=mode,
                  lon_st_slope=lon_st_slope)


def zdead_penalty(z_lon: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Hinge on the raw longitudinal controls: mean over valid drafts of mean_i relu(z_lon_i)^2 (i = 6 controls).
    z_lon [T, K, 6] (or [B, 6]), valid [T, K] (or [B]) -> float32 scalar (0 without valid drafts)."""
    zd = (torch.relu(z_lon.float()) ** 2).reshape(-1, z_lon.shape[-1]).mean(-1)
    m = valid.reshape(-1).float()
    return (zd * m).sum() / torch.clamp(m.sum(), min=1.0)


def compute_loss(out: Dict, batch: Dict, terms_fn: Callable, weights: Dict[str, float], pos_weight: float,
                 mode: str = "A", use_ttc: bool = False, w_gate: float = 1.0, cfg: Optional[Dict] = None,
                 lon_st_slope: float = 0.0, w_zdead: float = 0.0):
    """-> (loss scalar, stats dict of floats, dec).  lon_st_slope / w_zdead: mode-A dead-zone options (module docstring;
    0 = run 1, the loss is then computed exactly as before)."""
    T, K = batch["tau0"].shape[:2]
    dec = decode_batch(out, batch, mode, lon_st_slope)
    terms = terms_fn(dec, batch, T, K, cfg, ttc_grad=bool(float(weights.get("ttc", 0.0))))  # w_ttc 0: logged only
    m = batch["draft_valid"].reshape(-1).float()
    den = torch.clamp(m.sum(), min=1.0)
    loss = out["gate_logit"].new_zeros((), dtype=torch.float32)
    stats = {}
    for n, v in terms.items():
        v = torch.nan_to_num(v.float(), nan=0.0, posinf=1e3)
        mv = (v * m).sum() / den
        stats[f"t_{n}"] = float(mv.detach())
        w = float(weights.get(n, 0.0))
        if w:
            loss = loss + w * mv
    y = gate_targets(batch["labels"], use_ttc).reshape(-1)
    bce = F.binary_cross_entropy_with_logits(out["gate_logit"].float().reshape(-1), y,
                                             pos_weight=torch.tensor(pos_weight, device=y.device), reduction="none")
    gl = (bce * m).sum() / den
    stats["gate_bce"] = float(gl.detach())
    stats["corr"] = float(loss.detach())
    loss = loss + w_gate * gl
    zd = zdead_penalty(out["z_lon"], batch["draft_valid"])
    stats["zdead"] = float(zd.detach())
    stats["loss_ex_zd"] = float(loss.detach())
    if w_zdead:
        loss = loss + float(w_zdead) * zd
    stats["loss"] = float(loss.detach())
    stats["n_drafts"] = float(m.sum())
    stats["alpha_mean"] = float(dec["flags"]["alpha"].detach().mean())
    live = (dec["c_lon"].detach() < 0).any(-1).float()                       # = decoder.lon_live(z_lon), per draft
    stats["live"] = float((live * m).sum() / den)
    return loss, stats, dec


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC with average ranks for ties (nan if one class is empty)."""
    y = np.asarray(y, bool)
    score = np.asarray(score, np.float64)
    if y.all() or (~y).all():
        return float("nan")
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    avg = (np.cumsum(cnt) - cnt) + (cnt + 1) / 2.0
    r = avg[inv]
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# ----------------------------------------------------------------------------------------------- setup
def code_hashes() -> Dict[str, str]:
    d = REPO / "navsim/agents/para_ssr/refiner"
    out = {}
    for f in CODE_FILES + ("__init__.py",):
        p = d / f
        if p.exists():
            out[f] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    out["train_refiner.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    return out


def parse_weights(spec: str) -> Dict[str, float]:
    """'col=1,...,ttc=1' -> DEFAULT_W with the given entries replaced (keys must be in TERMS)."""
    w = dict(DEFAULT_W)
    if spec:
        for kv in spec.split(","):
            k, v = kv.split("=")
            if k.strip() not in TERMS:
                raise SystemExit(f"--w: unknown term {k.strip()!r} (terms: {', '.join(TERMS)})")
            w[k.strip()] = float(v)
    return w


def run_name(a) -> str:
    return a.run or f"{a.tag}_{a.arm}_fold{a.fold}_seed{a.seed}"


def select_rows(packed: RD.PackedSplit, fold: int, inner_frac: float, limit: Optional[int] = None,
                subset_tokens=None):
    """-> (inner-train rows, inner-val rows).  subset_tokens (run 4 --token-subset): only rows whose token is in it;
    applied before the inner-val log selection.  None -> runs 1-3 behaviour exactly."""
    rows = packed.select(RD.TRAIN_PARTS, exclude_folds=None if fold < 0 else [fold])
    if subset_tokens is not None:
        rows = RD.restrict_rows(packed, rows, subset_tokens)
    logs = packed.index.log.values[rows]
    ival = RD.inner_val_logs(logs, inner_frac)
    is_val = np.isin(logs, list(ival))
    tr, va = rows[~is_val], rows[is_val]
    if limit is not None:
        tr, va = tr[:limit], va[:max(1, limit // 4)]
        if len(va) == 0 or len(tr) == 0:
            tr, va = rows[:limit], rows[:max(1, limit // 4)]
    return tr, va


def build_net(arm: str, seed: int, norm: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              map_norm: Optional[Tuple[np.ndarray, np.ndarray]] = None, **kw) -> RefinerNet:
    """norm: BEVFusion z-score (arms T, TM); map_norm: ReSMap z-score (arms M, TM)."""
    mean, std = (norm if norm is not None else (None, None))
    if map_norm is None:
        return RefinerNet(arm, mean, std, seed=seed, **kw)
    return RefinerNet(arm, mean, std, seed=seed, map_norm_mean=map_norm[0], map_norm_std=map_norm[1], **kw)


NORM_FILES = {"det": "norm.npz", "map": "norm_map.npz"}
ARM_BRANCHES = {"none": (), "T": ("det",), "M": ("map",), "TM": ("det", "map")}


def load_run_norms(run_dir: Path, arm: str):
    """-> (det norm | None, map norm | None) from <run>/norm.npz / norm_map.npz as the arm needs them."""
    out = {}
    for b in ARM_BRANCHES[arm]:
        mean, std, _ = load_norm(Path(run_dir) / NORM_FILES[b])
        out[b] = (mean, std)
    return out.get("det"), out.get("map")


def load_run_model(run_dir: Path, ckpt: str = "best", device="cpu") -> Tuple[RefinerNet, Dict]:
    """Rebuild the net of a run (config.json + norm.npz) and load ckpt_<ckpt>.pt -> (net.eval(), config)."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    norm, map_norm = load_run_norms(run_dir, cfg["arm"])
    net = build_net(cfg["arm"], cfg["seed"], norm, map_norm, **cfg.get("net_kw", {}))
    p = run_dir / f"ckpt_{ckpt}.pt"
    if not p.exists() and ckpt == "best":
        p = run_dir / "ckpt_last.pt"
    sd = torch.load(p, map_location="cpu")
    net.load_state_dict(sd["model"])
    return net.to(device).eval(), cfg


def _device(gpu: int) -> torch.device:
    if gpu is None or gpu < 0:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(f"--gpu {gpu} requested but CUDA is not available (CUDA_VISIBLE_DEVICES?)")
    return torch.device(f"cuda:{gpu}")


def _log(path: Path, rec: Dict):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


@torch.no_grad()
def evaluate(net, loader, dev, terms_fn, weights, pos_weight, a, max_batches: Optional[int] = None) -> Dict:
    net.eval()
    sur_cfg = surrogate_config(a)
    agg, n = {}, 0
    P, Y = [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = RD.batch_to(batch, dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=a.amp and dev.type == "cuda"):
            out = net(batch["bev"], batch["tau0"], batch["v0"], batch["a0"], batch["eds"], batch["cmd"])
        _, st, _ = compute_loss(out, batch, terms_fn, weights, pos_weight, a.mode, a.gate_ttc, a.w_gate, cfg=sur_cfg,
                                lon_st_slope=getattr(a, "lon_st_slope", 0.0), w_zdead=getattr(a, "w_zdead", 0.0))
        w = st["n_drafts"]
        for k, v in st.items():
            agg[k] = agg.get(k, 0.0) + (v * w if k != "n_drafts" else v)
        n += w
        m = batch["draft_valid"].reshape(-1).cpu().numpy()
        P.append(out["gate_logit"].float().reshape(-1).cpu().numpy()[m])
        Y.append(gate_targets(batch["labels"], a.gate_ttc).reshape(-1).cpu().numpy()[m])
    res = {k: (v / max(n, 1.0) if k != "n_drafts" else v) for k, v in agg.items()}
    if P:
        res["gate_auc"] = auc(np.concatenate(P), np.concatenate(Y))
    net.train()
    return res


# ----------------------------------------------------------------------------------------------- main
def check_dead_zone_args(a) -> None:
    """--lon-st-slope / --w-zdead: finite, >= 0; the straight-through slope exists only in mode A."""
    for n in ("lon_st_slope", "w_zdead"):
        v = float(getattr(a, n))
        if not (math.isfinite(v) and v >= 0.0):
            raise SystemExit(f"--{n.replace('_', '-')} must be finite and >= 0, got {v}")
    if a.lon_st_slope and a.mode != "A":
        raise SystemExit("--lon-st-slope applies to the mode-A clamp only (decoder ignores it in mode B)")


def check_ttc_args(a) -> None:
    """--m-ttc finite if given; the TTC weight finite and >= 0."""
    m = getattr(a, "m_ttc", None)
    if m is not None and not math.isfinite(float(m)):
        raise SystemExit(f"--m-ttc must be finite, got {m}")
    w = parse_weights(a.w)["ttc"]
    if not (math.isfinite(w) and w >= 0.0):
        raise SystemExit(f"--w ttc=... must be finite and >= 0, got {w}")


def check_resume_config(run: Path, a) -> None:
    """Refuse (SystemExit, non-zero) to reuse a run directory whose config.json was written with other surrogate
    margins (m_col / m_dac), dead-zone options (lon_st_slope / w_zdead) or TTC options (m_ttc / weights['ttc']).
    Called BEFORE the finished-run (DONE)
    early exit, so a finished run launched with other options is an error, not a silent success (which would let
    stageT_gpu_commands.sh go on and overwrite that run's eval_* outputs).  A config.json without the dead-zone keys
    (run 1) means 0 / 0; without m_col / m_dac / m_ttc means unset; without weights['ttc'] means 0.  No-op on --restart
    or when there is no config.json."""
    if a.restart or not (run / "config.json").exists():
        return
    old = json.loads((run / "config.json").read_text())
    if (old.get("m_col"), old.get("m_dac")) != (getattr(a, "m_col", None), getattr(a, "m_dac", None)):
        raise SystemExit(f"{run}: config.json has m_col={old.get('m_col')} m_dac={old.get('m_dac')}, this call "
                         f"m_col={a.m_col} m_dac={a.m_dac}; use another --tag / TAG or --restart")
    o_dz = (float(old.get("lon_st_slope", 0.0)), float(old.get("w_zdead", 0.0)))
    if o_dz != (float(a.lon_st_slope), float(a.w_zdead)):
        raise SystemExit(f"{run}: config.json has lon_st_slope={o_dz[0]} w_zdead={o_dz[1]}, this call "
                         f"lon_st_slope={a.lon_st_slope} w_zdead={a.w_zdead}; use another --tag / TAG or --restart")
    o_ttc = (old.get("m_ttc"), float((old.get("weights") or {}).get("ttc", 0.0)))
    n_ttc = (getattr(a, "m_ttc", None), float(parse_weights(a.w)["ttc"]))
    if o_ttc != n_ttc:
        raise SystemExit(f"{run}: config.json has m_ttc={o_ttc[0]} w_ttc={o_ttc[1]}, this call m_ttc={n_ttc[0]} "
                         f"w_ttc={n_ttc[1]}; use another --tag / TAG or --restart")
    for key, arg in (("token_subset", "token_subset"), ("dev_token_subset", "dev_token_subset")):
        o_sha = (old.get(key) or {}).get("sha256")
        p = getattr(a, arg, None)
        n_sha = RD.load_token_subset(p)[1]["sha256"] if p else None
        if o_sha != n_sha:
            raise SystemExit(f"{run}: config.json has {key} sha256={o_sha}, this call {n_sha} ({p}); use another "
                             f"--tag / TAG or --restart")


def load_subsets(a):
    """-> (train subset tokens | None, train subset info | None, dev subset info | None); arms M / TM need a train
    subset whose tokens are all in the ReSMap cache of the split (checked in train())."""
    tok = info = dinfo = None
    if getattr(a, "token_subset", None):
        tok, info = RD.load_token_subset(a.token_subset)
    if getattr(a, "dev_token_subset", None):
        dinfo = RD.load_token_subset(a.dev_token_subset)[1]
    if a.arm in ("M", "TM") and tok is None:
        raise SystemExit(f"--arm {a.arm} needs --token-subset (tokens covered by the ReSMap cache, e.g. "
                         f"splits/train_trainlogs.parquet)")
    return tok, info, dinfo


def check_resmap_coverage(mp, tokens, what: str) -> None:
    miss = [t for t in tokens if not mp.has(str(t))]
    if miss:
        raise SystemExit(f"{what}: {len(miss)} of {len(tokens)} tokens are not in the ReSMap cache {mp.root} "
                         f"(e.g. {miss[:3]})")


def train(a) -> Path:
    check_dead_zone_args(a)
    check_ttc_args(a)
    sub_tok, sub_info, dev_info = load_subsets(a)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    dev = _device(a.gpu)
    run = Path(a.runs) / run_name(a)
    run.mkdir(parents=True, exist_ok=True)
    check_resume_config(run, a)                          # before the DONE exit: a finished run with other options fails
    if (run / "DONE").exists() and not a.restart:
        print(f"{run} is finished (DONE); --restart to retrain")
        return run
    if a.restart:
        for f in ("ckpt_last.pt", "ckpt_best.pt", "DONE", "log.jsonl", "summary.json"):
            (run / f).unlink(missing_ok=True)
    packed = RD.PackedSplit(a.split, a.packed_root)
    tr_rows, va_rows = select_rows(packed, a.fold, a.inner_val_frac, a.limit_tokens, subset_tokens=sub_tok)
    if len(tr_rows) == 0:
        raise RuntimeError(f"no trainable rows in {packed.dir} (fold {a.fold})")
    map_cache = None
    if a.arm in ("T", "none"):                                    # runs 1-3 code path, unchanged
        teacher = RD.TeacherCache.for_subset(RD.SPLIT_SUBSET[a.split]) if a.arm == "T" else None
        if a.teacher_root and teacher is not None:
            teacher = RD.TeacherCache(a.teacher_root)
        det_cache = teacher
    else:
        teacher, det_cache, map_cache = RD.arm_teachers(a.arm, a.split, a.teacher_root, a.resmap_root)
        check_resmap_coverage(map_cache, sub_tok, f"--token-subset {a.token_subset}")
    norms = {}
    for b in ARM_BRANCHES[a.arm]:
        npath = run / NORM_FILES[b]
        if not npath.exists():
            toks = packed.index.token.values[tr_rows]
            mean, std, info = RD.compute_teacher_norm(det_cache if b == "det" else map_cache, toks, a.n_norm, seed=0)
            info.update(split=a.split, fold=a.fold)
            if b == "map" or sub_info is not None:
                info.update(branch=b, token_subset_sha256=(sub_info or {}).get("sha256"))
            save_norm(npath, mean, std, info)
        mean, std, ninfo = load_norm(npath)
        norms[b] = (mean, std)
    norm, map_norm = norms.get("det"), norms.get("map")
    weights = parse_weights(a.w)
    terms_fn, sur_label = resolve_surrogate(a.surrogate)
    sur_cfg = surrogate_config(a)
    if sur_cfg is not None and sur_label == "stub":
        raise SystemExit("--m-col / --m-dac / --m-ttc need the real surrogate (the stub ignores them)")
    if weights["ttc"] and sur_label == "stub":
        raise SystemExit("--w ttc=... needs the real surrogate (the stub has no TTC term)")
    pi = label_prevalence(packed, tr_rows, a.gate_ttc)
    pos_weight = float(min((1 - pi) / pi, POS_WEIGHT_CAP)) if 0 < pi < 1 else 1.0
    net_kw = dict(dropout=a.dropout)
    net = build_net(a.arm, a.seed, norm, map_norm, **net_kw).to(dev)
    extra = {}
    if sub_info is not None:
        n_all = len(packed.select(RD.TRAIN_PARTS))
        extra["token_subset"] = dict(sub_info, n_packed_rows=int(len(RD.restrict_rows(
            packed, packed.select(RD.TRAIN_PARTS), sub_tok))), n_packed_rows_all=int(n_all),
            n_train_rows=int(len(tr_rows)), n_ival_rows=int(len(va_rows)))
    if dev_info is not None:
        extra["dev_token_subset"] = dev_info
    if map_cache is not None:
        extra.update(resmap_root=str(map_cache.root), resmap_sha_head=map_cache.sha_head,
                     norm_files={NORM_FILES[b]: (det_cache if b == "det" else map_cache).sha_head
                                 for b in ARM_BRANCHES[a.arm]})
    cfg = dict(vars(a), run=run_name(a), run_dir=str(run), surrogate=sur_label,
               surrogate_cfg=surrogate_config_record(sur_cfg, sur_label), weights=weights, pi=pi,
               pos_weight=pos_weight, n_train_tokens=int(len(tr_rows)), n_ival_tokens=int(len(va_rows)),
               param_counts=net.param_counts(), code=code_hashes(), packed=str(packed.dir), net_kw=net_kw,
               teacher_root=str(teacher.root) if teacher is not None else None,
               teacher_sha_head=teacher.sha_head if teacher is not None else None, device=str(dev), **extra)
    if not (run / "config.json").exists() or a.restart:
        (run / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    if sur_label == "stub":
        print("WARNING: using the STUB surrogate -- smoke / plumbing only")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    tl = RD.make_loader(packed, tr_rows, teacher, a.tokens_per_batch, shuffle=True, seed=a.seed, workers=a.workers)
    vl = RD.make_loader(packed, va_rows, teacher, a.tokens_per_batch, shuffle=False, seed=0, workers=a.workers,
                        drop_last=False)
    steps_per_epoch = max(1, len(tl))
    total = steps_per_epoch * a.epochs
    if a.max_steps:
        total = min(total, a.max_steps)
    warm = int(a.warmup_frac * total)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    use_amp = a.amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, step, best, bad = 0, 0, float("inf"), 0
    last = run / "ckpt_last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu")
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, step, best, bad = ck["epoch"] + 1, ck["step"], ck["best"], ck["bad"]
        print(f"resumed from {last}: epoch {start_epoch}, step {step}, best {best:.5f}")
    logp = run / "log.jsonl"
    stop = False
    ep = start_epoch - 1
    for ep in range(start_epoch, a.epochs):
        tl.generator.manual_seed(a.seed * 100003 + ep)
        net.train()
        t0 = time.time()
        agg, nb = {}, 0
        live_n, live_d = 0.0, 0.0
        for batch in tl:
            batch = RD.batch_to(batch, dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                out = net(batch["bev"], batch["tau0"], batch["v0"], batch["a0"], batch["eds"], batch["cmd"])
            loss, st, _ = compute_loss(out, batch, terms_fn, weights, pos_weight, a.mode, a.gate_ttc, a.w_gate,
                                       cfg=sur_cfg, lon_st_slope=a.lon_st_slope, w_zdead=a.w_zdead)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if a.clip > 0:
                scaler.unscale_(opt)
                st["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip))
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            nb += 1
            for k, v in st.items():
                agg[k] = agg.get(k, 0.0) + v
            live_n += st["live"] * st["n_drafts"]
            live_d += st["n_drafts"]
            if step % a.log_every == 0:
                _log(logp, dict(kind="step", epoch=ep, step=step, lr=sched.get_last_lr()[0],
                                **{k: v for k, v in st.items()}, sec=round(time.time() - t0, 1)))
            if a.max_steps and step >= a.max_steps:
                stop = True
                break
        tr = {k: v / max(nb, 1) for k, v in agg.items()}
        va = evaluate(net, vl, dev, terms_fn, weights, pos_weight, a, a.max_val_batches)
        improved = va.get("loss", float("inf")) < best
        if improved:
            best, bad = va["loss"], 0
            torch.save({"model": net.state_dict(), "epoch": ep, "step": step, "val": va}, run / "ckpt_best.pt")
        else:
            bad += 1
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "step": step, "best": best, "bad": bad}, last)
        _log(logp, dict(kind="epoch", epoch=ep, step=step, sec=round(time.time() - t0, 1), improved=improved,
                        bad_epochs=bad, val_loss=va.get("loss"), val_liveness=va.get("live"),
                        train_liveness=(live_n / live_d if live_d > 0 else None),
                        train={k: round(v, 6) for k, v in tr.items()},
                        val={k: (round(v, 6) if isinstance(v, float) else v) for k, v in va.items()}))
        if bad >= a.patience:
            print(f"early stop at epoch {ep} (patience {a.patience})")
            break
        if stop:
            break
    summ = dict(run=run_name(a), best_val_loss=best, epochs_run=ep + 1,
                steps=step, finished=time.strftime("%Y-%m-%dT%H:%M:%S"), max_steps=a.max_steps)
    (run / "summary.json").write_text(json.dumps(summ, indent=1))
    if not a.max_steps:
        (run / "DONE").write_text(summ["finished"])
    return run


def get_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["T", "none", "M", "TM"], required=True)
    ap.add_argument("--fold", type=int, default=-1, help="held-out fold 0..4 (cross-fitting); -1 = train on all folds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=-1, help="CUDA device index; -1 = CPU")
    ap.add_argument("--split", default="train")
    ap.add_argument("--packed-root", default=str(RD.DATA_ROOT / "packed"))
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--run", default=None, help="run name (default <tag>_<arm>_fold<k>_seed<s>)")
    ap.add_argument("--tag", default="stageT")
    ap.add_argument("--teacher-root", default=None, help="override the teacher cache root (manifest-checked)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--tokens-per-batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers (<= 2)")
    ap.add_argument("--amp", type=int, default=1, help="fp16 autocast on GPU")
    ap.add_argument("--mode", choices=["A", "B"], default="A")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--w", default="", help="loss weights, e.g. col=1,dac=1,prog=2,cmf=0.1,mod=0.1[,ttc=1] (ttc default 0)")
    ap.add_argument("--w-gate", type=float, default=DEFAULT_W_GATE)
    ap.add_argument("--gate-ttc", type=int, default=0, help="gate label also counts TTC < 1")
    ap.add_argument("--surrogate", choices=["auto", "real", "stub"], default="real",
                    help="real (default; fails if surrogate.py is unusable) | auto | stub (plumbing tests only)")
    ap.add_argument("--m-col", type=float, default=None,
                    help="surrogate collision margin [m]; default: SurrogateConfig() (code default, 0.3)")
    ap.add_argument("--m-dac", type=float, default=None,
                    help="surrogate DAC margin [m]; default: SurrogateConfig() (code default, 0.2)")
    ap.add_argument("--m-ttc", type=float, default=None,
                    help="surrogate TTC margin [m] (AMENDMENT 4); default: SurrogateConfig() (code default, 0.0); the "
                         "term enters the loss only with --w ...,ttc=<w> (default weight 0)")
    ap.add_argument("--lon-st-slope", type=float, default=0.0,
                    help="mode-A dead-zone fix (training losses only): straight-through backward slope lam of the "
                         "clamp c <- min(c, 0) on c > 0; the forward is unchanged; 0 = off (run 1)")
    ap.add_argument("--w-zdead", type=float, default=0.0,
                    help="weight of the hinge penalty mean(relu(z_lon)^2) on the raw lon controls (valid drafts); "
                         "0 = off (run 1)")
    ap.add_argument("--inner-val-frac", type=float, default=0.1)
    ap.add_argument("--n-norm", type=int, default=2048)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke: stop after this many steps (no DONE marker)")
    ap.add_argument("--max-val-batches", type=int, default=None)
    ap.add_argument("--limit-tokens", type=int, default=None, help="smoke: use only the first N train rows")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--token-subset", default=None,
                    help="run 4: token parquet (column token); packed rows are restricted to it before the inner-val "
                         "selection; recorded (sha256) in config.json; required for arms M / TM")
    ap.add_argument("--dev-token-subset", default=None,
                    help="run 4: dev token parquet recorded in config.json as eval_refiner's default dev subset")
    ap.add_argument("--resmap-root", default=None, help="override the ReSMap cache root (meta-checked; tests)")
    return ap


def main(argv=None):
    a = get_parser().parse_args(argv)
    if a.workers > 2:
        raise SystemExit("--workers must be <= 2 (shared machine)")
    train(a)


if __name__ == "__main__":
    main()
