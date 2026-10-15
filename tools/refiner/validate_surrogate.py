#!/usr/bin/env python
"""M8: surrogate (surrogate.py, M7) vs OFFICIAL labels on dev drafts -- IMPL_SPEC §3.7 acceptance test.

No pilot draft bank existed when this was written, so the validation set is built here from dev tokens that E already
cached (dev = held-out navtrain logs; E's 9,000 navtrain tokens have a metric cache; 2,173 of them are dev):
  pools per token (all official labels from tools/refiner/score_trajectories.score_token, bitwise = pdm_score):
    bank    13 drafts of decoder.sample_bank (k = 0 identity = HUMAN, 1-12 perturbation families, mode A bank)
    student E's PARA-SSR student draft (dump ego_traj) of the token (real draft)
    erule   E's rule-based corrections of the student draft (S_mix, S_cv, T_cv, T50_cv, G; only where they modified it)
    guided  surrogate-optimised corrections: decode(tau0, z, w, v0, 'A') with (z, w) = best iterate of 60 Adam steps
            (lr 0.05) on surrogate_loss (DEFAULT_WEIGHTS, no gate) from 0, for the 12 perturbed bank drafts + the
            student draft -> what training the refiner on this surrogate pushes drafts towards
    random  decode(tau0, z ~ N(0, 0.5), w ~ N(0, 0.3), mode 'A') of the same 13 sources (sha256(token) seeded)
  Unmodified outputs (max |traj - tau0| <= 1e-6) keep the source bytes and scores and are not pairs.
  Surrogate columns are computed on the raw 41-point reference (the loss) and, as a diagnostic ceiling, on the official
  LQR-tracked states (score_token(return_states=True)) -- the gap between the two is what M7b could recover.

Pre-stated definitions (written before any M8 number was seen)
  surrogate flag (collision, "at the training margin")  col_viol > 0 <=> some counted (object, n) has g < m_col = 0.3 m
  surrogate flag at margin 0                            col_hard     <=> some counted pair overlaps (exact SAT)
  official NC failure                                   nc < 1  (0 agent / 0.5 static at-fault collision)
  ACCEPTANCE (IMPL_SPEC §3.7), all on the dev bank / guided pools:
   A1 NC recall >= 0.70     : P(col flag | nc < 1) over bank drafts k = 0..12
   A2 human false alarm <= 2%: P(col flag | human identity draft k = 0, nc == 1)
   A3 pair P(official fixed | surrogate says fixed) >= 0.6, on guided pairs (tau0 = source, tau1 = its correction):
        surrogate says fixed = col flag(tau0) and not col flag(tau1);  official fixed = nc(tau0) < 1 and nc(tau1) == 1
      (literal reading; also reported: conditioning on nc(tau0) < 1, on both, the random / erule pools, DAC and NC u DAC)
  Reported, not in the acceptance: AUC (C_col, and max violation m_col - gmin), recall at FPR 1 / 5 %, precision / FPR,
  strata (agent vs static NC, first collision time < / >= 2 s, family), time (|first_n - nc_time_idx| <= 5) and object
  (first_obj track == nc_track) agreement, DAC recall / FPR / human FA at m_dac = 0.2 and 0, masks' effect (no human
  mask, no behind mask, + stopped-ego mask), tracked-state ceiling, progress P vs official raw progress and shapely,
  EP surrogate vs official EP, keyframe comfort vs official comfort, UNKNOWN rates, gradient checks (finite; a small
  normalised step along -grad C_col at z = w = 0 never raises C_col and the finite-difference directional derivative
  matches -|grad|), log-cluster bootstrap (2,000) and Clopper-Pearson CIs.

Result (800 tokens, 212 logs, 32,269 trajectories; details in the JSON): A1 0.948 PASS, A2 0.0275 FAIL, A3 0.510 FAIL.
Post-hoc decomposition (labelled as such in the JSON, "posthoc_guided_NC") is not part of the acceptance.

Subcommands
  select    --n 800 --seed 0       dev & E-cached & SDF & no frame gap -> <out>/tokens.parquet (token, log, frame_idx)
            (then build objects: build_future_objects.py --tokens <out>/tokens.parquet --logs <trainval> \
                                   --out /home/external-user/ssd/yongjae_refiner/objects/dev --workers 2)
  run       --workers 2 [--limit N] resumable shards <out>/shards/part-*.parquet (one row per trajectory)
  summarize                         -> report/refiner_T/surrogate_validation.json (+ printed summary)
CPU only; torch 1 thread per worker; <= 2 workers (shared machine).
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

import numpy as np
import pandas as pd

ROOT = Path("/home/external-user/yongjae/SSR")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/refiner"))

DATA = Path("/home/external-user/ssd/yongjae_refiner")
OUT = DATA / "m8"
OBJ_DIR = DATA / "objects/dev"
HUMAN = DATA / "human/dev.npz"
HUMAN_TRAIN = DATA / "human/train.npz"
SPLIT = DATA / "splits/dev.parquet"
E_DIR = ROOT / "report/cause_and_correction_tests/E_train_split_feasibility"
E_DUMP = E_DIR / "dump/npz"
E_CORR = E_DIR / "corrected_targets.npz"
E_VARIANTS = ("S_mix", "S_cv", "T_cv", "T50_cv", "G")
REPORT = ROOT / "report/refiner_T/surrogate_validation.json"
FAM_STUDENT = 20
FAM_ERULE = {v: 21 + i for i, v in enumerate(E_VARIANTS)}
GUIDED_STEPS, GUIDED_LR = 60, 0.05
GRAD_EPS = 1e-4                   # step length of the gradient sign check (in (z, w) units)
RAND_Z_SD, RAND_W_SD = 0.5, 0.3
MOD_EPS = 1e-6
OFF_KEYS = ("nc", "dac", "ddc", "ep", "ttc", "comfort", "pdms", "mult", "raw_progress", "pdm_progress_eff",
            "nc_track", "nc_time_idx", "nc_obj_type", "dac_time_idx", "ttc_time_idx")


# ----------------------------------------------------------------------------------------------- select
def cmd_select(a):
    from navsim.agents.para_ssr.refiner import sdf as S
    s = pd.read_parquet(SPLIT)
    s = s[s.e_cached].copy()
    h = np.load(HUMAN)
    hidx = {t: i for i, t in enumerate(h["tokens"].tolist())}
    s = s[[t in hidx and not bool(h["frame_gap"][hidx[t]]) for t in s.token]]
    s = s[[S.sdf_path(t, "navtrain").exists() and (E_DUMP / f"{t}.npz").exists() for t in s.token]]
    rng = np.random.default_rng(a.seed)
    pick = s.iloc[np.sort(rng.choice(len(s), min(a.n, len(s)), replace=False))]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    pick[["token", "log", "frame_idx"]].reset_index(drop=True).to_parquet(Path(a.out) / "tokens.parquet")
    print(json.dumps(dict(eligible=len(s), picked=len(pick), logs=int(pick.log.nunique()), seed=a.seed)))


# ----------------------------------------------------------------------------------------------- worker
_W: dict = {}


def _init():
    import torch
    torch.set_num_threads(1)
    import score_trajectories as ST
    h = np.load(HUMAN)
    _W["h"] = {k: h[k] for k in ("traj", "path", "n_reg", "v0", "a0")}
    _W["hidx"] = {t: i for i, t in enumerate(h["tokens"].tolist())}
    _W["ecorr"] = np.load(E_CORR)
    _W["ekeys"] = set(_W["ecorr"].files)
    _W["sim"], _W["scorer"] = ST.get_simulator_scorer()


def _rng(token: str, salt: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(f"{salt}:{token}".encode()).hexdigest()[:16], 16))


def _states_to_n(states: np.ndarray, ra) -> np.ndarray:
    """official tracked states [K, 41, 11] (global) -> rear-axle poses [K, 41, 3] in N."""
    c, s = math.cos(ra.heading), math.sin(ra.heading)
    dx, dy = states[..., 0] - ra.x, states[..., 1] - ra.y
    h = np.unwrap(states[..., 2] - ra.heading, axis=-1)
    return np.stack([c * dx + s * dy, -s * dx + c * dy, h], -1)


def _guided(src, scene, v0):
    """surrogate-optimised mode-A corrections of src [M, 8, 3] (float64 tensor): Adam (GUIDED_STEPS, GUIDED_LR) on the
    per-draft surrogate loss from z = w = 0, returning each draft's BEST iterate (lowest loss; the identity if nothing
    beats it, which then decodes to the source bytes).  Also the step-0 gradient diagnostics:
      g_finite    all gradients finite over the run
      g_gnorm0    |grad C_col| at the identity (z, w)
      g_col_step  C_col after one step of length GRAD_EPS along -grad C_col / |grad C_col|
      g_dd_ratio  finite-difference directional derivative / (-|grad|) (1 = exact first order; < 1 where the mode-A clamp
                  makes the +z side one-sided)"""
    import torch
    from navsim.agents.para_ssr.refiner import decoder as D
    from navsim.agents.para_ssr.refiner import surrogate as SU
    M = src.shape[0]
    idx = torch.zeros(M, dtype=torch.long)
    v0t = torch.full((M,), float(v0), dtype=src.dtype)
    z = torch.zeros(M, 6, dtype=src.dtype, requires_grad=True)
    w = torch.zeros(M, 6, dtype=src.dtype, requires_grad=True)
    dec = D.decode(src, z, w, v0t, "A")
    t0 = SU.surrogate_terms(dec, src, scene, idx)
    gz, gw = torch.autograd.grad(t0["col"].sum(), [z, w])
    finite = (torch.isfinite(gz).all(1) & torch.isfinite(gw).all(1)).numpy()
    gn = torch.cat([gz, gw], 1).norm(dim=1)
    with torch.no_grad():
        st = GRAD_EPS / torch.clamp(gn, min=1e-300)
        ds = D.decode(src, -gz * st[:, None], -gw * st[:, None], v0t, "A")
        col_step = SU.collision_cost(ds["dense"], scene, idx, details=False)["cost"]
        dd = (col_step - t0["col"].detach()) / GRAD_EPS
        dd_ratio = torch.where(gn > 0, dd / (-gn), torch.full_like(gn, float("nan")))
    opt = torch.optim.Adam([z, w], lr=GUIDED_LR)
    best = torch.full((M,), float("inf"), dtype=src.dtype)
    bz, bw = torch.zeros(M, 6, dtype=src.dtype), torch.zeros(M, 6, dtype=src.dtype)
    loss0 = None
    for it in range(GUIDED_STEPS + 1):
        dec = D.decode(src, z, w, v0t, "A")
        tot, terms = SU.surrogate_loss(dec, src, scene, idx)
        per = terms["per_draft"].detach()
        if it == 0:
            loss0 = per.numpy().copy()
        better = per < best
        best = torch.where(better, per, best)
        bz[better], bw[better] = z.detach()[better], w.detach()[better]
        if it == GUIDED_STEPS:
            break
        opt.zero_grad()
        (tot * M).backward()
        finite &= (torch.isfinite(z.grad).all(1) & torch.isfinite(w.grad).all(1)).numpy()
        opt.step()
    with torch.no_grad():
        dec = D.decode(src, bz, bw, v0t, "A")
        _, terms = SU.surrogate_loss(dec, src, scene, idx)
    diag = dict(g_loss0=loss0, g_loss1=terms["per_draft"].numpy(), g_col0=t0["col"].detach().numpy(),
                g_col1=terms["col"].numpy(), g_finite=finite, g_gnorm0=gn.numpy(), g_col_step=col_step.numpy(),
                g_dd_ratio=dd_ratio.numpy(), g_zmax=bz.abs().max(1).values.numpy(), g_wmax=bw.abs().max(1).values.numpy())
    return dec["traj"].detach(), dec, diag


def _random(src, token, v0):
    import torch
    from navsim.agents.para_ssr.refiner import decoder as D
    M = src.shape[0]
    rng = _rng(token, "m8_random_v1")
    z = torch.as_tensor(rng.normal(0, RAND_Z_SD, (M, 6)), dtype=src.dtype)
    w = torch.as_tensor(rng.normal(0, RAND_W_SD, (M, 6)), dtype=src.dtype)
    with torch.no_grad():
        dec = D.decode(src, z, w, torch.full((M,), float(v0), dtype=src.dtype), "A")
    return dec["traj"], dec


def _surrogate_cols(traj, scene, tracked, mc, cl_full_check=True):
    """surrogate columns for trajectories [K, 8, 3] (float64 tensor) + tracked N-frame poses [K, 41, 3]."""
    import torch
    from shapely.geometry import Point
    from navsim.agents.para_ssr.refiner import surrogate as SU
    K = traj.shape[0]
    idx = torch.zeros(K, dtype=torch.long)
    cfg = SU.SurrogateConfig()
    ev = SU.evaluate_trajectories(traj, scene, idx, cfg)
    col, dac = ev["col"], ev["dac"]
    tracks = scene.track[0]
    fo = col["first_obj"].numpy()
    cols = dict(
        col_cost=col["cost"].numpy(), col_viol=col["viol"].numpy(), col_gmin=col["gmin"].numpy(),
        col_gmin_hard=col["gmin_hard"].numpy(), col_hard=col["hard_overlap"].numpy(), col_first_n=col["first_n"].numpy(),
        col_first_track=np.array([str(tracks[j]) if j >= 0 else "" for j in fo]), col_nhm=col["n_human_masked"].numpy(),
        dac_cost=dac["cost"].numpy(), dac_viol=dac["viol"].numpy(), dac_min=dac["sdf_min"].numpy(),
        dac_hard=dac["hard_out"].numpy(), dac_first_n=dac["first_n"].numpy(), dac_oog=dac["n_oog"].numpy(),
        P_sur=ev["P"].numpy(), ep_sur=ev["ep_sur"].numpy(), unknown=ev["unknown"].numpy(),
        kf_ratio=ev["cmf"]["kf_ratio"].numpy(), kf_flag=ev["cmf"]["kf_flag"].numpy())
    dense = ev["dense"]
    for name, c in (("nohm", SU.SurrogateConfig(use_human_mask=False)), ("nobehind", SU.SurrogateConfig(behind_m=None)),
                    ("stop", SU.SurrogateConfig(min_speed=0.05))):
        o = SU.collision_cost(dense, scene, idx, c)
        cols[f"col_viol_{name}"] = o["viol"].numpy()
        cols[f"col_hard_{name}"] = o["hard_overlap"].numpy()
    cols["unknown_spec"] = SU.unknown_footprint(dense, scene, idx, SU.SurrogateConfig(radius_margin=None)).numpy()
    # tracked-state ceiling
    td = torch.as_tensor(tracked)
    tc = SU.collision_cost(td, scene, idx, cfg)
    tn = SU.collision_cost(td, scene, idx, SU.SurrogateConfig(use_human_mask=False))
    tdac = SU.dac_cost(td, scene, idx, cfg)
    cols.update(t_col_viol=tc["viol"].numpy(), t_col_hard=tc["hard_overlap"].numpy(),
                t_col_gmin_hard=tc["gmin_hard"].numpy(), t_col_hard_nohm=tn["hard_overlap"].numpy(),
                t_dac_viol=tdac["viol"].numpy(), t_dac_hard=tdac["hard_out"].numpy(), t_dac_min=tdac["sdf_min"].numpy())
    # progress: cropped torch projection vs shapely on the FULL metric-cache centerline (global frame)
    if cl_full_check:
        ra = mc.ego_state.rear_axle
        c_, s_ = math.cos(ra.heading), math.sin(ra.heading)
        cen = SU.ego_centre(dense[:, [0, -1]]).numpy()
        gx, gy = ra.x + c_ * cen[..., 0] - s_ * cen[..., 1], ra.y + s_ * cen[..., 0] + c_ * cen[..., 1]
        pr = np.array([mc.centerline.project([Point(gx[k, 0], gy[k, 0]), Point(gx[k, 1], gy[k, 1])]) for k in range(K)])
        cols["P_shp"] = np.clip(pr[:, 1] - pr[:, 0], 0, None)
    return cols


def process_token(task):
    import torch
    import score_trajectories as ST
    from navsim.agents.para_ssr.refiner import decoder as D
    from navsim.agents.para_ssr.refiner import gt_future as GF
    from navsim.agents.para_ssr.refiner import sdf as S
    from navsim.agents.para_ssr.refiner import surrogate as SU
    token, log = task
    t_start = time.time()
    try:
        mc = ST.load_metric_cache(ST.locate_metric_cache(token, log))
        objs = GF.load_objects(OBJ_DIR / f"{token}.npz")
        sdf = S.load_sdf(S.sdf_path(token, "navtrain"))
        cl = SU.centerline_from_metric_cache(mc)
        h, i = _W["h"], _W["hidx"][token]
        tau_h, v0, a0 = h["traj"][i], float(h["v0"][i]), float(h["a0"][i])
        bank = D.sample_bank(tau_h, token, path_long=h["path"][i], n_valid=int(h["n_reg"][i]), v0=v0, a0=a0,
                             centerline=cl)
        stu = np.load(E_DUMP / f"{token}.npz")["ego_traj"].astype(np.float32)
        er = [(v, _W["ecorr"][f"{v}__{token}"].astype(np.float32)) for v in E_VARIANTS if f"{v}__{token}" in _W["ekeys"]]
        # ---- set 1: bank + student + E corrections
        set1 = np.concatenate([bank["drafts"], stu[None]] + [x[None] for _, x in er], 0)
        meta = [dict(pool="bank", fam=int(bank["family"][k]), src=-1, bank_valid=bool(bank["valid"][k]))
                for k in range(13)]
        meta.append(dict(pool="student", fam=FAM_STUDENT, src=-1, bank_valid=True))
        meta += [dict(pool="erule", fam=FAM_ERULE[v], src=13, bank_valid=True) for v, _ in er]
        sc1 = ST.score_token(mc, set1, _W["sim"], _W["scorer"], return_states=True)
        p_pdm = sc1[0]["pdm_progress_eff"]
        scene = SU.collate_scenes([SU.scene_from_numpy(objs, sdf, cl, tau_h, p_pdm, v0, a0)])
        # ---- guided + random corrections of the 12 perturbed bank drafts + the student draft
        src_k = list(range(1, 13)) + [13]
        src = torch.as_tensor(set1[src_k].astype(np.float64))
        g_traj, g_dec, gd = _guided(src, scene, v0)
        r_traj, r_dec = _random(src, token, v0)
        rows_traj, rows_meta, dec_extra = list(set1), list(meta), [None] * len(set1)
        for pool, tr, dec in (("guided", g_traj, g_dec), ("random", r_traj, r_dec)):
            with torch.no_grad():
                terms = SU.surrogate_terms(dec, src, scene, torch.zeros(len(src_k), dtype=torch.long), details=True)
            for m, k in enumerate(src_k):
                t32 = tr[m].numpy().astype(np.float32)
                dev = float(np.abs(t32.astype(np.float64) - set1[k].astype(np.float64)).max())
                modified = dev > MOD_EPS
                rows_traj.append(t32 if modified else set1[k])
                x = dict(pool=pool, fam=meta[k]["fam"], src=k, bank_valid=meta[k]["bank_valid"], modified=modified,
                         max_dev=dev)
                ex = dict(an_ratio=float(terms["details"]["cmf"]["an_ratio"][m]),
                          an_flag=bool(terms["details"]["cmf"]["an_flag"][m]), c_mod=float(terms["mod"][m]),
                          l_prog=float(terms["prog"][m]), dec_alpha=float(dec["flags"]["alpha"][m]))
                if pool == "guided":
                    ex.update({kk: (vv[m].item() if hasattr(vv[m], "item") else vv[m]) for kk, vv in gd.items()})
                rows_meta.append(x)
                dec_extra.append(ex)
        # ---- set 2: score only the modified outputs
        n1 = len(set1)
        mod_i = [j for j in range(n1, len(rows_traj)) if rows_meta[j]["modified"]]
        sc_all = list(sc1) + [None] * (len(rows_traj) - n1)
        if mod_i:
            sc2 = ST.score_token(mc, np.stack([rows_traj[j] for j in mod_i]), _W["sim"], _W["scorer"], return_states=True)
            for j, r in zip(mod_i, sc2):
                sc_all[j] = r
        for j in range(n1, len(rows_traj)):
            if sc_all[j] is None:
                sc_all[j] = sc1[rows_meta[j]["src"]]
        # ---- surrogate on every trajectory (raw reference + tracked states)
        ra = mc.ego_state.rear_axle
        tracked = _states_to_n(np.stack([r["states"] for r in sc_all]), ra)
        cols = _surrogate_cols(torch.as_tensor(np.stack(rows_traj).astype(np.float64)), scene, tracked, mc)
        rows = []
        for j in range(len(rows_traj)):
            r = dict(token=token, log=log, k=j, **{kk: rows_meta[j].get(kk) for kk in ("pool", "fam", "src", "bank_valid")},
                     modified=bool(rows_meta[j].get("modified", False)), max_dev=float(rows_meta[j].get("max_dev", 0.0)))
            r.update({kk: sc_all[j][kk] for kk in OFF_KEYS})
            r.update({kk: (vv[j].item() if hasattr(vv[j], "item") else vv[j]) for kk, vv in cols.items()})
            if dec_extra[j] is not None:
                r.update(dec_extra[j])
            r["n_obj"] = int(objs["kf"].shape[0])
            r["error"] = ""
            rows.append(r)
        rows[0]["sec"] = time.time() - t_start
        return rows
    except Exception as e:  # noqa: BLE001
        import traceback
        return [dict(token=token, log=log, k=-1, error=f"{type(e).__name__}: {e} | {traceback.format_exc()[-800:]}")]


def cmd_run(a):
    from multiprocessing import Pool
    out = Path(a.out)
    shards = out / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    toks = pd.read_parquet(out / "tokens.parquet")
    have = [p for p in toks.token if (OBJ_DIR / f"{p}.npz").exists()]
    toks = toks[toks.token.isin(have)]
    done = set()
    for p in shards.glob("part-*.parquet"):
        d = pd.read_parquet(p, columns=["token", "error"])
        done |= set(d.token[d.error == ""]) if a.retry_errors else set(d.token)
    todo = toks[~toks.token.isin(done)]
    if a.limit:
        todo = todo.iloc[: a.limit]
    tasks = list(zip(todo.token, todo.log))
    print(f"[m8] {len(toks)} tokens with objects, {len(done)} done, {len(tasks)} to do, workers {a.workers}", flush=True)
    t0 = time.time()
    buf, n_done, part = [], 0, len(list(shards.glob("part-*.parquet")))

    def flush():
        nonlocal buf, part
        if buf:
            pd.DataFrame(buf).to_parquet(shards / f"part-{part:05d}.parquet")
            part += 1
            buf = []

    if a.workers <= 1:
        _init()
        it = map(process_token, tasks)
    else:
        pool = Pool(a.workers, initializer=_init)
        it = pool.imap_unordered(process_token, tasks, chunksize=1)
    for rows in it:
        buf.extend(rows)
        n_done += 1
        if n_done % a.shard_tokens == 0:
            flush()
            el = time.time() - t0
            print(f"[m8] {n_done}/{len(tasks)} {el:.0f}s  ETA {el / n_done * (len(tasks) - n_done) / 60:.1f} min",
                  flush=True)
    flush()
    if a.workers > 1:
        pool.close()
        pool.join()
    print(f"[m8] finished {n_done} tokens in {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------------------------- summarize
def _cp(k, n, alpha=0.05):
    from scipy.stats import beta
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def _rate(mask_num, mask_den):
    k, n = int((mask_num & mask_den).sum()), int(mask_den.sum())
    return dict(k=k, n=n, p=(k / n if n else float("nan")), ci=_cp(k, n))


def _boot(log, num, den, n=2000, seed=0):
    """log-cluster bootstrap 95% interval of sum(num) / sum(den): logs resampled with replacement (per-log sums)."""
    log = pd.Series(np.asarray(log))
    codes, uniq = pd.factorize(log)
    L = len(uniq)
    sn = np.bincount(codes, weights=np.asarray(num, float), minlength=L)
    sd = np.bincount(codes, weights=np.asarray(den, float), minlength=L)
    rng = np.random.default_rng(seed)
    cnt = rng.multinomial(L, np.full(L, 1.0 / L), size=n)                 # [n, L] times each log is drawn
    tn, td = cnt @ sn, cnt @ sd
    v = tn[td > 0] / td[td > 0]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, bool)
    s = np.nan_to_num(np.asarray(s, np.float64), nan=-1e9, posinf=1e9, neginf=-1e9)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def _recall_at_fpr(y, s, fpr):
    y = np.asarray(y, bool)
    s = np.nan_to_num(np.asarray(s, np.float64), nan=-1e9, posinf=1e9, neginf=-1e9)
    neg = np.sort(s[~y])[::-1]
    if len(neg) == 0 or y.sum() == 0:
        return float("nan")
    thr = neg[min(len(neg) - 1, int(np.floor(fpr * len(neg))))]
    return float((s[y] > thr).mean())


def _traj_block(d, y, flag, score, name):
    y, flag = np.asarray(y, bool), np.asarray(flag, bool)
    tp = int((y & flag).sum())
    return {"name": name, "n": int(len(y)), "pos": int(y.sum()), "flag": int(flag.sum()),
            "recall": _rate(flag, y), "precision": (tp / flag.sum() if flag.sum() else float("nan")),
            "fpr": _rate(flag, ~y), "auc": _auc(y, score), "recall_at_fpr1": _recall_at_fpr(y, score, 0.01),
            "recall_at_fpr5": _recall_at_fpr(y, score, 0.05)}


def _pairs(df, pool):
    """(source, modified) pairs of a pool -> DataFrame with *_0 (source) and *_1 (modified) columns."""
    src = df[df.pool.isin(["bank", "student"])].set_index(["token", "k"])
    p = df[(df.pool == pool) & df.modified].copy()
    s = src.loc[list(zip(p.token, p.src))].reset_index(drop=True)
    p = p.reset_index(drop=True)
    cols = ["nc", "dac", "ddc", "ttc", "comfort", "pdms", "ep", "col_viol", "col_cost", "dac_viol", "dac_cost",
            "col_hard", "dac_hard", "P_sur", "raw_progress", "fam", "t_col_viol", "t_col_hard", "t_dac_hard",
            "col_gmin_hard"]
    out = pd.DataFrame({"token": p.token, "log": p.log})
    for c in cols:
        out[c + "_0"] = s[c].to_numpy()
        out[c + "_1"] = p[c].to_numpy()
    for c in ("l_prog", "c_mod", "max_dev", "an_flag"):
        if c in p:
            out[c] = p[c].to_numpy()
    return out


def _pair_block(pp, fail0, fail1, flag0, flag1, dcost, name):
    fixed_s = flag0 & ~flag1
    fixed_o = fail0 & ~fail1
    res = {"name": name, "n_pairs": int(len(pp)),
           "A3_literal P(off fixed | sur fixed)": _rate(fixed_o, fixed_s),
           "P(off pass1 | off fail0, sur clean1)": _rate(~fail1, fail0 & ~flag1),
           "P(off pass1 | off fail0 & sur flag0, sur clean1)": _rate(~fail1, fail0 & flag0 & ~flag1),
           "P(off new fail | off pass0, sur clean1)": _rate(fail1, ~fail0 & ~flag1),
           "P(sur flag1 | off new fail)": _rate(flag1, ~fail0 & fail1),
           "off fixed": int(fixed_o.sum()), "off new fail": int((~fail0 & fail1).sum()),
           "sur fixed": int(fixed_s.sum()), "sur new flag": int((~flag0 & flag1).sum())}
    chg = fail0 != fail1
    if chg.any():
        agree = np.where(fail0 & ~fail1, dcost < 0, dcost > 0)[chg]
        res["sign agreement dcost vs official change"] = _rate(agree, np.ones(int(chg.sum()), bool))
    return res


def cmd_summarize(a):
    import torch
    from navsim.agents.para_ssr.refiner import decoder as D
    from navsim.agents.para_ssr.refiner import surrogate as SU
    out = Path(a.out)
    df = pd.concat([pd.read_parquet(p) for p in sorted((out / "shards").glob("part-*.parquet"))], ignore_index=True)
    err = df[df.error != ""]
    df = df[df.error == ""].copy()
    df.to_parquet(out / "rows.parquet")
    for c in ("col_viol", "col_viol_nohm", "col_viol_nobehind", "col_viol_stop", "t_col_viol", "dac_viol", "t_dac_viol"):
        df[c] = df[c].astype(float)
    df["fail_nc"] = df.nc < 1
    df["fail_dac"] = df.dac < 1
    df["fail_any"] = (df.nc < 1) | (df.dac < 1) | (df.ddc < 1)
    df["flag_col"] = df.col_viol > 0
    df["flag_dac"] = df.dac_viol > 0
    R = {"created": time.strftime("%F %T"), "n_tokens": int(df.token.nunique()), "n_logs": int(df.log.nunique()),
         "n_rows": int(len(df)), "n_errors": int(err.token.nunique()),
         "config": {k: getattr(SU.SurrogateConfig(), k) for k in SU.SurrogateConfig.__dataclass_fields__},
         "weights_guided": SU.DEFAULT_WEIGHTS, "guided": dict(steps=GUIDED_STEPS, lr=GUIDED_LR),
         "random": dict(z_sd=RAND_Z_SD, w_sd=RAND_W_SD)}
    bank = df[df.pool == "bank"]
    human = bank[bank.fam == 0]
    # ---------------- official label rates
    R["official_rates"] = {pool: {m: float((d[m] < 1).mean()) for m in ("nc", "dac", "ddc", "ttc", "comfort")} |
                           {"any_multi": float(d.fail_any.mean()), "n": int(len(d))}
                           for pool, d in df.groupby("pool")}
    R["bank_family_fail"] = {D.FAMILY_NAME.get(int(f), str(f)): {"n": int(len(d)), "nc": float(d.fail_nc.mean()),
                                                                 "dac": float(d.fail_dac.mean()),
                                                                 "any": float(d.fail_any.mean())}
                             for f, d in bank.groupby("fam")}
    # ---------------- acceptance
    A1 = _rate(bank.flag_col.to_numpy(), bank.fail_nc.to_numpy())
    A1["boot_log"] = _boot(bank.log, (bank.col_viol > 0) & (bank.nc < 1), bank.nc < 1)
    hs = human[human.nc == 1]
    A2 = _rate(hs.flag_col.to_numpy(), np.ones(len(hs), bool))
    A2["boot_log"] = _boot(hs.log, hs.col_viol > 0, np.ones(len(hs)))
    gp = _pairs(df, "guided")
    f0, f1 = gp.nc_0.to_numpy() < 1, gp.nc_1.to_numpy() < 1
    s0, s1 = gp.col_viol_0.to_numpy() > 0, gp.col_viol_1.to_numpy() > 0
    A3 = _rate(f0 & ~f1, s0 & ~s1)
    if len(gp):
        A3["boot_log"] = _boot(gp.log, f0 & ~f1 & s0 & ~s1, s0 & ~s1)
    R["acceptance"] = {
        "A1 NC recall >= 0.70 (bank, col flag at m_col)": A1 | {"pass": bool(A1["p"] >= 0.70)},
        "A2 human false alarm <= 0.02 (col flag at m_col)": A2 | {"pass": bool(A2["p"] <= 0.02)},
        "A3 pair P(official fixed | surrogate fixed) >= 0.6 (guided, literal)": A3 | {"pass": bool(A3["p"] >= 0.6)},
    }
    # ---------------- trajectory level
    tl = {}
    for pool, d in [("bank", bank), ("bank_perturbed", bank[bank.fam != 0]), ("student", df[df.pool == "student"]),
                    ("guided_out", df[(df.pool == "guided") & df.modified]),
                    ("random_out", df[(df.pool == "random") & df.modified]), ("all", df)]:
        if not len(d):
            continue
        y = d.fail_nc.to_numpy()
        tl[pool] = {
            "nc_spec": _traj_block(d, y, d.col_viol > 0, d.col_cost, "col flag m_col (spec)"),
            "nc_margin0": _traj_block(d, y, d.col_hard, d.col_viol, "col hard overlap (margin 0)"),
            "nc_maxviol_auc": _auc(y, d.col_viol),
            "nc_no_human_mask": _traj_block(d, y, d.col_viol_nohm > 0, d.col_viol_nohm, "no human mask"),
            "nc_no_behind_mask": _traj_block(d, y, d.col_viol_nobehind > 0, d.col_viol_nobehind, "no behind mask"),
            "nc_stop_mask": _traj_block(d, y, d.col_viol_stop > 0, d.col_viol_stop, "+ stopped-ego mask"),
            "nc_tracked_ceiling": _traj_block(d, y, d.t_col_viol > 0, d.t_col_viol, "tracked states, m_col"),
            "nc_tracked_margin0": _traj_block(d, y, d.t_col_hard, d.t_col_viol, "tracked states, margin 0"),
            "dac_spec": _traj_block(d, d.fail_dac.to_numpy(), d.dac_viol > 0, d.dac_cost, "dac flag m_dac"),
            "dac_margin0": _traj_block(d, d.fail_dac.to_numpy(), d.dac_hard, d.dac_viol, "dac margin 0"),
            "dac_tracked_margin0": _traj_block(d, d.fail_dac.to_numpy(), d.t_dac_hard, d.t_dac_viol, "tracked dac 0"),
            "any_multi_flag": _traj_block(d, d.fail_any.to_numpy(), (d.col_viol > 0) | (d.dac_viol > 0),
                                          d.col_cost + d.dac_cost, "col or dac flag vs NC|DAC|DDC"),
        }
    R["trajectory_level"] = tl
    # strata of NC recall on the bank
    fb = bank[bank.fail_nc]
    R["nc_recall_strata_bank"] = {
        "agent (nc=0)": _rate((fb.col_viol > 0).to_numpy(), (fb.nc == 0).to_numpy()),
        "static (nc=0.5)": _rate((fb.col_viol > 0).to_numpy(), (fb.nc == 0.5).to_numpy()),
        "first collision < 2 s": _rate((fb.col_viol > 0).to_numpy(), (fb.nc_time_idx < 20).to_numpy()),
        "first collision >= 2 s": _rate((fb.col_viol > 0).to_numpy(), (fb.nc_time_idx >= 20).to_numpy()),
        "by family": {D.FAMILY_NAME.get(int(f), str(f)): _rate((g.col_viol > 0).to_numpy(), np.ones(len(g), bool))
                      for f, g in fb.groupby("fam")},
        "miss: no raw overlap even without masks (col_hard_nohm False)":
            _rate((~fb.col_hard_nohm & ~(fb.col_viol > 0)).to_numpy(), np.ones(len(fb), bool)),
        "miss explained by human mask (flag without it)":
            _rate(((fb.col_viol_nohm > 0) & ~(fb.col_viol > 0)).to_numpy(), np.ones(len(fb), bool)),
        "miss explained by behind mask": _rate(((fb.col_viol_nobehind > 0) & ~(fb.col_viol > 0)).to_numpy(),
                                               np.ones(len(fb), bool)),
    }
    fl = fb[fb.col_viol > 0]
    R["nc_time_object_agreement_bank"] = {
        "|first_n - nc_time_idx| <= 5": _rate((np.abs(fl.col_first_n - fl.nc_time_idx) <= 5).to_numpy(),
                                              np.ones(len(fl), bool)),
        "median first_n - nc_time_idx": float(np.median(fl.col_first_n - fl.nc_time_idx)) if len(fl) else float("nan"),
        "first object == nc_track": _rate((fl.col_first_track == fl.nc_track).to_numpy(), np.ones(len(fl), bool)),
    }
    # human false alarms (all variants)
    R["human_false_alarm"] = {
        "n_human_nc_pass": int(len(hs)),
        "col m_col (spec)": _rate((hs.col_viol > 0).to_numpy(), np.ones(len(hs), bool)),
        "col margin 0": _rate(hs.col_hard.to_numpy(), np.ones(len(hs), bool)),
        "col m_col, no human mask": _rate((hs.col_viol_nohm > 0).to_numpy(), np.ones(len(hs), bool)),
        "col human pairs removed by the mask (tokens)": _rate((hs.col_nhm > 0).to_numpy(), np.ones(len(hs), bool)),
        "dac m_dac 0.2": _rate((human[human.dac == 1].dac_viol > 0).to_numpy(), np.ones(int((human.dac == 1).sum()), bool)),
        "dac margin 0": _rate(human[human.dac == 1].dac_hard.to_numpy(), np.ones(int((human.dac == 1).sum()), bool)),
        "human official fail rates": {m: float((human[m] < 1).mean()) for m in ("nc", "dac", "ddc", "ttc", "comfort")},
    }
    # ---------------- pairs
    pr = {}
    for pool in ("guided", "random", "erule"):
        pp = _pairs(df, pool)
        if not len(pp):
            continue
        f0, f1 = pp.nc_0.to_numpy() < 1, pp.nc_1.to_numpy() < 1
        s0, s1 = pp.col_viol_0.to_numpy() > 0, pp.col_viol_1.to_numpy() > 0
        d0, d1 = pp.dac_0.to_numpy() < 1, pp.dac_1.to_numpy() < 1
        q0, q1 = pp.dac_viol_0.to_numpy() > 0, pp.dac_viol_1.to_numpy() > 0
        pr[pool] = {
            "NC": _pair_block(pp, f0, f1, s0, s1, (pp.col_cost_1 - pp.col_cost_0).to_numpy(), "NC vs col"),
            "DAC": _pair_block(pp, d0, d1, q0, q1, (pp.dac_cost_1 - pp.dac_cost_0).to_numpy(), "DAC vs dac"),
            "NC|DAC": _pair_block(pp, f0 | d0, f1 | d1, s0 | q0, s1 | q1,
                                  (pp.col_cost_1 + pp.dac_cost_1 - pp.col_cost_0 - pp.dac_cost_0).to_numpy(), "NC|DAC"),
            "new failures any (nc/dac/ddc/ttc/comfort)": int((((pp.nc_0 == 1) & (pp.nc_1 < 1)) | ((pp.dac_0 == 1) & (pp.dac_1 < 1))
                                                              | ((pp.ddc_0 == 1) & (pp.ddc_1 < 1)) | ((pp.ttc_0 == 1) & (pp.ttc_1 < 1))
                                                              | ((pp.comfort_0 == 1) & (pp.comfort_1 < 1))).sum()),
            "mean dPDMS": float((pp.pdms_1 - pp.pdms_0).mean()),
            "mean dPDMS on NC-failing sources": float((pp.pdms_1 - pp.pdms_0)[f0].mean()) if f0.any() else float("nan"),
        }
        if "l_prog" in pp:
            dep = (pp.ep_0 - pp.ep_1).to_numpy()
            ok = (pp.nc_0 == 1).to_numpy() & (pp.nc_1 == 1).to_numpy() & (pp.dac_0 == 1).to_numpy() & (pp.dac_1 == 1).to_numpy()
            if ok.sum() > 2:
                pr[pool]["L_prog vs official EP drop (mult=1 both)"] = {
                    "n": int(ok.sum()), "spearman": float(pd.Series(pp.l_prog.to_numpy()[ok]).corr(pd.Series(dep[ok]), "spearman")),
                    "mae": float(np.abs(pp.l_prog.to_numpy()[ok] - dep[ok]).mean())}
    R["pairs"] = pr
    # ---------------- post-hoc decomposition (labelled; NOT part of the pre-stated acceptance)
    gp = _pairs(df, "guided")
    if len(gp):
        f0, f1 = gp.nc_0.to_numpy() < 1, gp.nc_1.to_numpy() < 1
        s0, s1 = gp.col_viol_0.to_numpy() > 0, gp.col_viol_1.to_numpy() > 0
        sf = s0 & ~s1
        t0, t1 = gp.ttc_0.to_numpy() < 1, gp.ttc_1.to_numpy() < 1
        h0, h1 = gp.col_hard_0.to_numpy(), gp.col_hard_1.to_numpy()
        res = f1 & ~s1                                          # official NC failures the raw surrogate calls clean
        R["posthoc_guided_NC"] = {
            "surrogate-fixed pairs by official state of tau0": {
                "nc fail": int((sf & f0).sum()), "nc pass & ttc fail": int((sf & ~f0 & t0).sum()),
                "nc pass & ttc pass": int((sf & ~f0 & ~t0).sum())},
            "... of the nc-pass & ttc-fail ones, ttc fixed at tau1": _rate(~t1, sf & ~f0 & t0),
            "... tau0 min exact SAT gap of the nc-pass ones [m] median / p90": (
                [float(np.median(gp.col_gmin_hard_0[sf & ~f0])), float(np.percentile(gp.col_gmin_hard_0[sf & ~f0], 90))]
                if (sf & ~f0).any() else None),
            "literal rule with the margin-0 flag (col_hard) instead of m_col": _rate(f0 & ~f1, h0 & ~h1),
            "official NC failures after guided correction": int(f1.sum()),
            "... raw surrogate clean (m_col) there": _rate(res, f1),
            "... of those, tracked-state surrogate flags them (m_col / margin 0)": [
                _rate(gp.t_col_viol_1.to_numpy() > 0, res), _rate(gp.t_col_hard_1.to_numpy(), res)],
        }
    # ---------------- guided diagnostics / gradients
    g = df[df.pool == "guided"]
    if len(g):
        act = (g.g_col0 > 1e-6) & (g.g_gnorm0 > 0)
        ga = g[act]
        R["gradients"] = {
            "n_sources": int(len(g)), "all_finite": bool(g.g_finite.all()), "finite_rate": float(g.g_finite.mean()),
            "max_grad_norm_at_identity": float(g.g_gnorm0.max()),
            "n sources with C_col > 1e-6": int(act.sum()),
            "step %.0e along -grad C_col: C_col lower" % GRAD_EPS: _rate((ga.g_col_step < ga.g_col0).to_numpy(), np.ones(len(ga), bool)),
            "... unchanged": _rate((ga.g_col_step == ga.g_col0).to_numpy(), np.ones(len(ga), bool)),
            "... higher": _rate((ga.g_col_step > ga.g_col0).to_numpy(), np.ones(len(ga), bool)),
            "fd directional derivative / -|grad|: median, p10, p90": [float(ga.g_dd_ratio.median()),
                                                                     float(ga.g_dd_ratio.quantile(0.1)),
                                                                     float(ga.g_dd_ratio.quantile(0.9))],
            "guided loss <= identity loss": _rate((g.g_loss1 <= g.g_loss0 + 1e-12).to_numpy(), np.ones(len(g), bool)),
            "guided modified": _rate(g.modified.to_numpy(), np.ones(len(g), bool)),
            "guided max_dev [m] median / p90 (modified)": [float(g[g.modified].max_dev.median()),
                                                           float(g[g.modified].max_dev.quantile(0.9))],
            "C_col>1e-6 sources: C_col reduced by > 50%": _rate((ga.g_col1 < 0.5 * ga.g_col0).to_numpy(),
                                                                 np.ones(len(ga), bool)),
        }
    # ---------------- progress / EP / comfort / unknown
    raw = df[~df.pool.isin(["guided", "random"]) | df.modified]
    ok = raw.mult == 1
    R["progress"] = {
        "P_sur vs shapely full line: max |diff| [m]": float(np.abs(raw.P_sur - raw.P_shp).max()),
        "P_sur vs shapely: n |diff| > 1e-6": int((np.abs(raw.P_sur - raw.P_shp) > 1e-6).sum()),
        "P_sur (reference) vs official raw_progress (tracked), mult=1": {
            "n": int(ok.sum()), "median_abs": float(np.median(np.abs(raw.P_sur - raw.raw_progress)[ok])),
            "p90_abs": float(np.percentile(np.abs(raw.P_sur - raw.raw_progress)[ok], 90)),
            "pearson": float(np.corrcoef(raw.P_sur[ok], raw.raw_progress[ok])[0, 1])},
        "ep_sur vs official EP, mult=1": {"mae": float(np.abs(raw.ep_sur - raw.ep)[ok].mean()),
                                          "p90_abs": float(np.percentile(np.abs(raw.ep_sur - raw.ep)[ok], 90)),
                                          "exact_1_agree": float(((raw.ep_sur == 1) == (raw.ep == 1))[ok].mean())},
    }
    R["comfort"] = {
        "keyframe flag vs official comfort<1 (bank)": _traj_block(bank, (bank.comfort < 1).to_numpy(), bank.kf_flag,
                                                                  bank.kf_ratio, "kf flag"),
        "keyframe flag vs official comfort<1 (all)": _traj_block(df, (df.comfort < 1).to_numpy(), df.kf_flag, df.kf_ratio,
                                                                 "kf flag"),
        "human keyframe flag rate (M8 dev tokens)": float(human.kf_flag.mean()),
        "human_tables": comfort_human_rates(),
    }
    R["unknown"] = {pool: {"extended (R-10 m)": float(d.unknown.mean()), "spec": float(d.unknown_spec.mean()), "n": int(len(d))}
                    for pool, d in df.groupby("pool")}
    R["objects"] = {"n_obj_mean": float(df.groupby("token").n_obj.first().mean()),
                    "n_obj_max": int(df.groupby("token").n_obj.first().max()),
                    "dac_oog_rows": int((df.dac_oog > 0).sum())}
    R["timing_s_per_token"] = {"mean": float(df.sec.dropna().mean()), "p90": float(df.sec.dropna().quantile(0.9))}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(R, indent=1, default=float))
    print(json.dumps(R["acceptance"], indent=1, default=float))
    print(f"-> {REPORT}")


def comfort_human_rates():
    """Human comfort-surrogate violation rates (keyframe terms at f = 0.9 and 1.0, analytic terms at identity) on the
    stage-T human tables (train + dev, frame-gap tokens excluded)."""
    import torch
    from navsim.agents.para_ssr.refiner import decoder as D
    from navsim.agents.para_ssr.refiner import surrogate as SU
    out = {}
    for name, path in (("dev", HUMAN), ("train", HUMAN_TRAIN)):
        if not path.exists():
            continue
        z = np.load(path)
        ok = ~z["frame_gap"]
        tr, v0 = torch.as_tensor(z["traj"][ok]), torch.as_tensor(z["v0"][ok])
        kf = SU.keyframe_comfort(tr, v0)
        per = {}
        for f in (0.9, 1.0):
            per[f"kf f={f}"] = {k: float(SU.comfort_penalty({k: x}, f)["flag"].float().mean()) for k, x in kf.items()}
            per[f"kf f={f}"]["any"] = float(SU.comfort_penalty(kf, f)["flag"].float().mean())
        an = []
        for s in range(0, len(tr), 4096):
            dec = D.decode(tr[s:s + 4096], torch.zeros(len(tr[s:s + 4096]), 6), torch.zeros(len(tr[s:s + 4096]), 6),
                           v0[s:s + 4096], "A")
            an.append(SU.comfort_penalty(SU.analytic_comfort(dec), SU.CMF_FRAC_AN)["flag"])
        an = torch.cat(an)
        used = SU.comfort_penalty(kf, SU.CMF_FRAC_KF)["flag"] | an
        per["analytic f=0.9 any"] = float(an.float().mean())
        per["USED (kf f=%.1f + analytic f=%.1f) any" % (SU.CMF_FRAC_KF, SU.CMF_FRAC_AN)] = float(used.float().mean())
        per["n"] = int(len(tr))
        out[name] = per
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--n", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(OUT))
    p = sub.add_parser("run")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard-tokens", type=int, default=25)
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--out", default=str(OUT))
    p = sub.add_parser("summarize")
    p.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    if getattr(a, "workers", 1) > 2:
        raise SystemExit("<= 2 workers (shared machine)")
    {"select": cmd_select, "run": cmd_run, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
