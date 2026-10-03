#!/usr/bin/env python
"""M8 re-check on the TRAIN split: surrogate margins (m_col, m_dac) x human-overlap mask rule (STATUS.md D1).

Pre-stated selection rule: report/refiner_T/m8_recheck/SELECTION_RULE.txt (written before any number of this run).
Reads ONLY stage-T train-split data (splits/train.parquet, drafts/train, scores/train.parquet, objects/train, human/train,
navtrain SDF, metric caches).  Dev tokens and navtest are never read (guarded in cmd_select / process_token).

Per token (one token per log):
  bank    the stored 13-draft bank (drafts/train/<token>.npz): k = 0 human identity + the VALID perturbed drafts k = 1..12
          (invalid slots are human bytes with valid = False -> skipped); re-scored officially with LQR-tracked states
          (return_states) and checked against the stored labels (column lab_ok).
  guided  for every grid setting s = (m_col, m_dac, mask): validate_surrogate._guided's procedure with config s
          (60 Adam steps, lr 0.05, surrogate_loss(DEFAULT_WEIGHTS, cfg_s), best iterate, mode A) on the valid perturbed
          bank drafts.  All settings of one mask rule are optimised in ONE batch with per-draft margin tensors
          (SurrogateConfig(m_col=[B,1,1], m_dac=[B,1,1]) -- valid because every per-draft term is independent and Adam
          is element-wise; tests/test_m8_recheck.py checks equality with separate scalar-config runs and with
          validate_surrogate._guided).  Unmodified outputs (max dev <= 1e-6 m) keep the source bytes and scores.
  surrogate columns for every trajectory (margin-free statistics, so every m_col / m_dac is a threshold):
          g_s_<mask>   min smooth separation g over counted (object, n)    (col flag at m  <=> g_s < m)
          g_h_<mask>   min exact SAT gap over counted (object, n)
          dmin         min SDF over valid corners                          (dac flag at m  <=> dmin < m)
          t_*          the same on the official LQR-tracked states (ideal-M7b diagnostic ceiling)
          <mask> in {sat: exact SAT human-overlap mask (current default), none: no human mask}

Subcommands
  select    --n 800 --seed 0     -> <out>/tokens.parquet (token, log, frame_idx)
  run       --workers 2 [--limit N]  resumable shards <out>/shards/part-*.parquet
  summarize                      -> report/refiner_T/m8_recheck/{m8_recheck.json, grid_table.csv, grid_tracked.csv}
CPU only; torch 1 thread per worker; <= 2 workers (shared machine).
"""
from __future__ import annotations

import argparse
import itertools
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
OUT = DATA / "m8_recheck"
REPORT_DIR = ROOT / "report/refiner_T/m8_recheck"
SPLIT = DATA / "splits/train.parquet"
HUMAN = DATA / "human/train.npz"
OBJ_DIR = DATA / "objects/train"
DRAFT_DIR = DATA / "drafts/train"
LABELS = DATA / "scores/train.parquet"

M_COLS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)
M_DACS = (0.0, 0.05, 0.1, 0.2)
MASKS = ("sat", "none")                   # sat = exact SAT human-overlap mask (current default); none = no mask
SETTINGS = [dict(sid=i, m_col=mc, m_dac=md, mask=mk)
            for i, (mk, mc, md) in enumerate(itertools.product(MASKS, M_COLS, M_DACS))]
DRAFTS_PER_CHUNK_OBJ = 60_000             # guided batch size cap: drafts x objects per optimisation chunk
NEAR_G, NEAR_SLACK = 8.0, 8.0             # optimisation-only object prefilter [m] (see near_objects)
SCORE_CHUNK = 64                          # trajectories per score_token call (+ PDM-Closed)
MOD_EPS = 1e-6
LAB_KEYS = ("nc", "dac", "ddc", "ep", "ttc", "comfort", "pdms")
OFF_KEYS = ("nc", "dac", "ddc", "ep", "ttc", "comfort", "pdms", "mult", "raw_progress", "pdm_progress_eff",
            "nc_track", "nc_time_idx", "nc_obj_type", "dac_time_idx", "ttc_time_idx")

# pre-stated thresholds (SELECTION_RULE.txt)
A1_MIN, A2_MAX, A3_MIN, DAC_HFA_MAX = 0.70, 0.02, 0.60, 0.02
MASK_PREF = {"sat": 1, "none": 0}


def _train_only(path: Path):
    s = str(path)
    if "/dev" in s or "navtest" in s:
        raise SystemExit(f"refusing to read a non-train path: {s}")


# ----------------------------------------------------------------------------------------------- select
def eligible_tokens() -> pd.DataFrame:
    """train-split tokens with drafts + stored labels (13 rows, no error) + objects + SDF + human (no frame gap) + mc."""
    import score_trajectories as ST
    from navsim.agents.para_ssr.refiner import sdf as S
    for p in (SPLIT, HUMAN, OBJ_DIR, DRAFT_DIR, LABELS):
        _train_only(p)
    s = pd.read_parquet(SPLIT)
    assert set(s.split.unique()) == {"train"}, s.split.unique()
    lab = pd.read_parquet(LABELS, columns=["token", "k", "error"])
    ok = lab.groupby("token").agg(n=("k", "size"), err=("error", lambda e: (e.astype(str) != "").any()))
    good = set(ok.index[(ok.n == 13) & ~ok.err])
    h = np.load(HUMAN)
    gap = dict(zip(h["tokens"].tolist(), h["frame_gap"].tolist()))
    s = s[s.token.isin(good) & s.token.map(lambda t: gap.get(t, True) is False)]
    s = s[[(OBJ_DIR / f"{t}.npz").exists() and (DRAFT_DIR / f"{t}.npz").exists()
           and S.sdf_path(t, "navtrain").exists() for t in s.token]]
    s = s[[ST.locate_metric_cache(t, l) is not None for t, l in zip(s.token, s.log)]]
    return s


def pick_one_per_log(s: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    s = s.sort_values(["log", "token"]).reset_index(drop=True)
    pick = [g.index[rng.integers(len(g))] for _, g in s.groupby("log", sort=True)]
    p = s.loc[pick].reset_index(drop=True)
    if len(p) > n:
        p = p.iloc[np.sort(rng.choice(len(p), n, replace=False))].reset_index(drop=True)
    return p


def cmd_select(a):
    s = eligible_tokens()
    p = pick_one_per_log(s, a.n, a.seed)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    p[["token", "log", "frame_idx"]].to_parquet(Path(a.out) / "tokens.parquet")
    # snapshot of the stored labels of the pool (scores/train.parquet is re-merged by the running draft-bank job)
    lab = pd.read_parquet(LABELS, columns=["token", "k", *LAB_KEYS, "family"])
    lab[lab.token.isin(set(p.token))].reset_index(drop=True).to_parquet(Path(a.out) / "labels_pool.parquet")
    print(json.dumps(dict(eligible=len(s), eligible_logs=int(s.log.nunique()), picked=len(p),
                          logs=int(p.log.nunique()), seed=a.seed)))


# ----------------------------------------------------------------------------------------------- guided (batched)
def guided_multi(src, scene, v0, m_col, m_dac, use_human_mask: bool, steps: int = None, lr: float = None):
    """validate_surrogate._guided with PER-DRAFT margins: src [M, 8, 3] float64 tensor, m_col / m_dac [M] (tensor or
    array).  Returns (traj [M, 8, 3] of the best iterate, best per-draft loss [M]).  With all margins equal to the
    defaults and use_human_mask=True this is exactly _guided's trajectory output."""
    import torch
    import validate_surrogate as V
    from navsim.agents.para_ssr.refiner import decoder as D
    from navsim.agents.para_ssr.refiner import surrogate as SU
    steps = V.GUIDED_STEPS if steps is None else steps
    lr = V.GUIDED_LR if lr is None else lr
    M = src.shape[0]
    dt = src.dtype
    mc_t = torch.as_tensor(np.asarray(m_col, np.float64), dtype=dt).reshape(M, 1, 1)
    md_t = torch.as_tensor(np.asarray(m_dac, np.float64), dtype=dt).reshape(M, 1, 1)
    cfg = SU.SurrogateConfig(m_col=mc_t, m_dac=md_t, use_human_mask=bool(use_human_mask))
    idx = torch.zeros(M, dtype=torch.long)
    v0t = torch.full((M,), float(v0), dtype=dt)
    z = torch.zeros(M, 6, dtype=dt, requires_grad=True)
    w = torch.zeros(M, 6, dtype=dt, requires_grad=True)
    opt = torch.optim.Adam([z, w], lr=lr)
    best = torch.full((M,), float("inf"), dtype=dt)
    bz, bw = torch.zeros(M, 6, dtype=dt), torch.zeros(M, 6, dtype=dt)
    for it in range(steps + 1):
        dec = D.decode(src, z, w, v0t, "A")
        tot, terms = SU.surrogate_loss(dec, src, scene, idx, cfg)
        per = terms["per_draft"].detach()
        better = per < best
        best = torch.where(better, per, best)
        bz[better], bw[better] = z.detach()[better], w.detach()[better]
        if it == steps:
            break
        opt.zero_grad()
        (tot * M).backward()
        opt.step()
    with torch.no_grad():
        dec = D.decode(src, bz, bw, v0t, "A")
    return dec["traj"].detach(), best


# ----------------------------------------------------------------------------------------------- speed-ups (exact)
def near_objects(scene, src, g_min: float = NEAR_G, slack: float = NEAR_SLACK):
    """bool [A]: objects that can matter for the guided optimisation of drafts src [M, 8, 3].

    Dropped iff at every dense time where it is OBS its centre is at distance D >= slack + sqrt(2) (g_min + r_obj +
    r_ego) from EVERY point of EVERY source dense reference (time-agnostic, because mode-A slowing changes the time
    alignment; r = half diagonals).  Any pose the decoder can reach lies within `slack` = 8 m of a source reference
    point (lateral |d| <= decoder.D_MAX = 2 m, 0.1 s point spacing <= 2 m at <= 40 m/s, heading change of the box
    centre <= RA2C * pi/2 ~ 2.3 m), and the SAT gap satisfies g_hard >= |d| / sqrt(2) - r_ego - r_obj (the ego long /
    lat axes are orthogonal), so a dropped object has g >= g_min - T ln 4 > 7.9 m from every reachable pose: its cost
    term beta softplus((m_col - g) / beta) <= 0.1 exp(-76) ~ 1e-34 and its gradient ~1e-33 (far below Adam's eps
    1e-8 times anything).  Used ONLY for the optimisation; every reported surrogate statistic uses the full scene.
    tests/test_m8_recheck.py checks identical guided outputs with and without the filter on real tokens.  (The bound
    covers the object times 0..4 s of the weighted terms; C_ttc, weight 0 in DEFAULT_WEIGHTS, is not in the guided loss.)"""
    import torch
    from navsim.agents.para_ssr.refiner import surrogate as SU
    from navsim.agents.para_ssr.refiner.geometry import HALF_LEN, HALF_WID, dense_reference
    with torch.no_grad():
        e = SU.ego_centre(dense_reference(src)).reshape(-1, 2)                  # [M*41, 2]
        b = scene.boxes[0]                                                      # [A, 41, 5]
        r_obj = 0.5 * torch.sqrt(b[..., 3] ** 2 + b[..., 4] ** 2)                # [A, 41]
        r_ego = float(np.hypot(HALF_LEN, HALF_WID))
        d = torch.cdist(b[..., :2].reshape(-1, 2), e.to(b.dtype)).min(1).values.reshape(b.shape[:2])
        near = (d < slack + math.sqrt(2.0) * (g_min + r_obj + r_ego)) & scene.obs[0].bool()
        return near.any(1)


def sub_scene(scene, keep):
    """SceneBatch restricted to objects keep [A] (single-scene batch)."""
    import dataclasses
    k = keep.nonzero().reshape(-1)
    tr = None if scene.track is None else [None if t is None else np.asarray(t)[k.numpy()] for t in scene.track]
    sub = lambda x: None if x is None else x[:, k]                          # noqa: E731
    return dataclasses.replace(scene, boxes=scene.boxes[:, k], obs=scene.obs[:, k], is_agent=scene.is_agent[:, k],
                               human_overlap=sub(scene.human_overlap), track=tr,
                               boxes_ttc=sub(getattr(scene, "boxes_ttc", None)),
                               obs_ttc=sub(getattr(scene, "obs_ttc", None)))


# ----------------------------------------------------------------------------------------------- surrogate stats
def surrogate_stats(dense, scene, prefix: str = "") -> dict:
    """margin-free per-trajectory statistics of dense references / tracked poses [K, 41, 3] (float64)."""
    import torch
    from navsim.agents.para_ssr.refiner import surrogate as SU
    K = dense.shape[0]
    idx = torch.zeros(K, dtype=torch.long)
    out = {}
    with torch.no_grad():
        for mk in MASKS:
            o = SU.collision_cost(dense, scene, idx, SU.SurrogateConfig(use_human_mask=(mk == "sat")))
            out[f"{prefix}g_s_{mk}"] = o["gmin"].numpy()
            out[f"{prefix}g_h_{mk}"] = o["gmin_hard"].numpy()
        d = SU.dac_cost(dense, scene, idx, SU.SurrogateConfig())
        out[f"{prefix}dmin"] = d["sdf_min"].numpy()
        out[f"{prefix}d_oog"] = d["n_oog"].numpy()
    return out


# ----------------------------------------------------------------------------------------------- worker
_W: dict = {}


def _init(pool_tokens=None, labels_path=None):
    import torch
    torch.set_num_threads(1)
    import score_trajectories as ST
    _train_only(HUMAN)
    _train_only(LABELS)
    h = np.load(HUMAN)
    _W["h"] = {k: h[k] for k in ("traj", "v0", "a0", "frame_gap")}
    _W["hidx"] = {t: i for i, t in enumerate(h["tokens"].tolist())}
    lab = pd.read_parquet(labels_path or LABELS, columns=["token", "k", *LAB_KEYS, "family"])
    if pool_tokens is not None:
        lab = lab[lab.token.isin(set(pool_tokens))]
    _W["lab"] = {t: g.sort_values("k") for t, g in lab.groupby("token")}
    _W["sim"], _W["scorer"] = ST.get_simulator_scorer()


def _score(mc, trajs):
    import score_trajectories as ST
    out = []
    for s in range(0, len(trajs), SCORE_CHUNK):
        out += ST.score_token(mc, trajs[s:s + SCORE_CHUNK], _W["sim"], _W["scorer"], return_states=True)
    return out


def process_token(task):
    import torch
    import score_trajectories as ST
    import validate_surrogate as V
    from navsim.agents.para_ssr.refiner import gt_future as GF
    from navsim.agents.para_ssr.refiner import sdf as S
    from navsim.agents.para_ssr.refiner import surrogate as SU
    from navsim.agents.para_ssr.refiner.geometry import dense_reference
    token, log = task
    t_start = time.time()
    try:
        dpath = DRAFT_DIR / f"{token}.npz"
        _train_only(dpath)
        with np.load(dpath) as z:
            drafts = np.asarray(z["drafts"], np.float32)
            fam = np.asarray(z["family"]).astype(int)
            valid = np.asarray(z["valid"]).astype(bool)
            assert str(z["split"]) == "train", str(z["split"])
        mc = ST.load_metric_cache(ST.locate_metric_cache(token, log))
        objs = GF.load_objects(OBJ_DIR / f"{token}.npz")
        sdf = S.load_sdf(S.sdf_path(token, "navtrain"))
        cl = SU.centerline_from_metric_cache(mc)
        h, i = _W["h"], _W["hidx"][token]
        assert not bool(h["frame_gap"][i])
        tau_h, v0, a0 = h["traj"][i], float(h["v0"][i]), float(h["a0"][i])
        bank_k = [0] + [k for k in range(1, 13) if valid[k]]
        src_k = bank_k[1:]
        # ---- bank: official re-score with tracked states; stored-label check
        sc_bank = _score(mc, drafts[bank_k])
        lab = _W["lab"][token].set_index("k")
        lab_ok = [all(sc_bank[j][c] == lab.loc[k, c] for c in LAB_KEYS) for j, k in enumerate(bank_k)]
        p_pdm = sc_bank[0]["pdm_progress_eff"]
        scene = SU.collate_scenes([SU.scene_from_numpy(objs, sdf, cl, tau_h, p_pdm, v0, a0)])
        A = max(1, int(objs["kf"].shape[0]))
        # ---- guided corrections for every setting (batched per mask rule, chunked by drafts x objects)
        src = torch.as_tensor(drafts[src_k].astype(np.float64))
        M = len(src_k)
        g_rows = []                                       # (setting, src k, traj float32, modified, max_dev, loss)
        t_g = time.time()
        n_near, hov_any = 0, bool(scene.human_overlap is not None and scene.human_overlap.any())
        if M:
            keep = near_objects(scene, src)
            n_near = int(keep.sum())
            scene_opt = sub_scene(scene, keep)
            A_opt = max(1, n_near)
            # no human-overlap pair anywhere -> the two mask rules are the same loss: optimise once, reuse
            mask_runs = MASKS if hov_any else ("sat",)
            res = {}
            for mk in mask_runs:
                sets = [s for s in SETTINGS if s["mask"] == mk]
                per_chunk = max(1, DRAFTS_PER_CHUNK_OBJ // (A_opt * M))
                for c0 in range(0, len(sets), per_chunk):
                    ss = sets[c0:c0 + per_chunk]
                    srcb = src.repeat(len(ss), 1, 1)
                    mcb = np.repeat([s["m_col"] for s in ss], M)
                    mdb = np.repeat([s["m_dac"] for s in ss], M)
                    tr, loss = guided_multi(srcb, scene_opt, v0, mcb, mdb, mk == "sat")
                    tr = tr.numpy()
                    for q, s in enumerate(ss):
                        res[(mk, s["m_col"], s["m_dac"])] = (tr[q * M:(q + 1) * M], loss[q * M:(q + 1) * M])
            for s in SETTINGS:
                tr, loss = res[(s["mask"] if hov_any else "sat", s["m_col"], s["m_dac"])]
                for m, k in enumerate(src_k):
                    t32 = tr[m].astype(np.float32)
                    dev = float(np.abs(t32.astype(np.float64) - drafts[k].astype(np.float64)).max())
                    g_rows.append((s, k, t32 if dev > MOD_EPS else drafts[k], dev > MOD_EPS, dev, float(loss[m])))
        t_g = time.time() - t_g
        # ---- unique modified outputs -> official scores (+ tracked states)
        uniq, key2u = [], {}
        for r in g_rows:
            if r[3]:
                kb = r[2].tobytes()
                if kb not in key2u:
                    key2u[kb] = len(uniq)
                    uniq.append(r[2])
        t_s = time.time()
        sc_u = _score(mc, np.stack(uniq)) if uniq else []
        t_s = time.time() - t_s
        # ---- surrogate statistics on raw references and tracked states
        all_tr = np.concatenate([drafts[bank_k]] + ([np.stack(uniq)] if uniq else []), 0)
        all_sc = list(sc_bank) + list(sc_u)
        ra = mc.ego_state.rear_axle
        tracked = V._states_to_n(np.stack([r["states"] for r in all_sc]), ra)
        st = {}
        CH = max(1, DRAFTS_PER_CHUNK_OBJ // A)
        for c0 in range(0, len(all_tr), CH):
            dense = dense_reference(torch.as_tensor(all_tr[c0:c0 + CH].astype(np.float64)))
            a_ = surrogate_stats(dense, scene)
            b_ = surrogate_stats(torch.as_tensor(tracked[c0:c0 + CH]), scene, "t_")
            for kk, vv in {**a_, **b_}.items():
                st.setdefault(kk, []).append(vv)
        st = {kk: np.concatenate(vv) for kk, vv in st.items()}

        def row(j_all, **extra):
            r = dict(token=token, log=log, **extra)
            r.update({kk: all_sc[j_all][kk] for kk in OFF_KEYS})
            r.update({kk: float(vv[j_all]) for kk, vv in st.items()})
            return r

        rows = []
        for j, k in enumerate(bank_k):
            rows.append(row(j, pool="bank", k=k, fam=int(fam[k]), sid=-1, m_col=np.nan, m_dac=np.nan, mask="",
                            modified=False, max_dev=0.0, loss=np.nan, lab_ok=bool(lab_ok[j])))
        nb = len(bank_k)
        for s, k, t32, mod, dev, loss in g_rows:
            j = nb + key2u[t32.tobytes()] if mod else bank_k.index(k)
            rows.append(row(j, pool="guided", k=k, fam=int(fam[k]), sid=s["sid"], m_col=s["m_col"],
                            m_dac=s["m_dac"], mask=s["mask"], modified=bool(mod), max_dev=dev, loss=loss,
                            lab_ok=True))
        for r in rows:
            r["n_obj"] = int(objs["kf"].shape[0])
            r["error"] = ""
        rows[0].update(sec=time.time() - t_start, sec_guided=t_g, sec_score=t_s, n_unique=len(uniq), n_near=n_near,
                       human_overlap_any=hov_any)
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
    done = set()
    for p in shards.glob("part-*.parquet"):
        d = pd.read_parquet(p, columns=["token", "error"])
        done |= set(d.token[d.error == ""]) if a.retry_errors else set(d.token)
    todo = toks[~toks.token.isin(done)]
    if a.limit:
        todo = todo.iloc[: a.limit]
    tasks = list(zip(todo.token, todo.log))
    print(f"[m8r] {len(toks)} tokens, {len(done)} done, {len(tasks)} to do, workers {a.workers}", flush=True)
    t0 = time.time()
    buf, n_done, part = [], 0, len(list(shards.glob("part-*.parquet")))

    def flush():
        nonlocal buf, part
        if buf:
            pd.DataFrame(buf).to_parquet(shards / f"part-{part:05d}.parquet")
            part += 1
            buf = []

    pool_tokens = list(toks.token)
    if a.workers <= 1:
        _init(pool_tokens, out / "labels_pool.parquet")
        it = map(process_token, tasks)
    else:
        pool = Pool(a.workers, initializer=_init, initargs=(pool_tokens, out / "labels_pool.parquet"))
        it = pool.imap_unordered(process_token, tasks, chunksize=1)
    for rows in it:
        buf.extend(rows)
        n_done += 1
        if n_done % a.shard_tokens == 0:
            flush()
            el = time.time() - t0
            print(f"[m8r] {n_done}/{len(tasks)} {el:.0f}s  ETA {el / n_done * (len(tasks) - n_done) / 60:.1f} min",
                  flush=True)
    flush()
    if a.workers > 1:
        pool.close()
        pool.join()
    print(f"[m8r] finished {n_done} tokens in {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------------------------- summarize
def _stat(num, den, log):
    import validate_surrogate as V
    num, den = np.asarray(num, bool), np.asarray(den, bool)
    r = V._rate(num, den)
    r["boot_log"] = V._boot(np.asarray(log)[den], num[den], np.ones(int(den.sum()))) if den.any() else [np.nan] * 2
    return r


def _flag(d, kind: str, m: float, mask: str, tracked: bool = False):
    p = "t_" if tracked else ""
    if kind == "col":
        return (d[f"{p}g_s_{mask}"] < m).to_numpy()
    return (d[f"{p}dmin"] < m).to_numpy()


def setting_metrics(df: pd.DataFrame, s: dict, tracked: bool = False) -> dict:
    """pre-stated metrics of setting s (flags on the raw reference, or on tracked states if tracked)."""
    mc, md, mk = s["m_col"], s["m_dac"], s["mask"]
    bank = df[df.pool == "bank"]
    human = bank[bank.k == 0]
    nc_fail = (bank.nc < 1).to_numpy()
    dac_fail = (bank.dac < 1).to_numpy()
    fcol = _flag(bank, "col", mc, mk, tracked)
    fdac = _flag(bank, "dac", md, mk, tracked)
    hs = (human.nc == 1).to_numpy()
    hd = (human.dac == 1).to_numpy()
    out = {"A1": _stat(fcol, nc_fail, bank.log), "A2": _stat(_flag(human, "col", mc, mk, tracked), hs, human.log),
           "DAC_recall": _stat(fdac, dac_fail, bank.log), "DAC_fpr": _stat(fdac, ~dac_fail, bank.log),
           "DAC_human_FA": _stat(_flag(human, "dac", md, mk, tracked), hd, human.log),
           "NC_fpr": _stat(fcol, ~nc_fail, bank.log)}
    # pairs: guided(s) modified outputs vs their bank source
    g = df[(df.pool == "guided") & (df.sid == s["sid"])]
    src = bank.set_index(["token", "k"])
    gm = g[g.modified]
    s0 = src.loc[list(zip(gm.token, gm.k))]
    f0, f1 = (s0.nc < 1).to_numpy(), (gm.nc < 1).to_numpy()
    c0, c1 = _flag(s0, "col", mc, mk, tracked), _flag(gm, "col", mc, mk, tracked)
    d0, d1 = (s0.dac < 1).to_numpy(), (gm.dac < 1).to_numpy()
    q0, q1 = _flag(s0, "dac", md, mk, tracked), _flag(gm, "dac", md, mk, tracked)
    sf = c0 & ~c1
    t0 = (s0.ttc < 1).to_numpy()
    out["A3"] = _stat(f0 & ~f1, sf, gm.log)
    out["A3_decomp"] = {"sur fixed": int(sf.sum()), "tau0 nc fail": int((sf & f0).sum()),
                        "tau0 nc pass & ttc fail": int((sf & ~f0 & t0).sum()),
                        "tau0 nc pass & ttc pass": int((sf & ~f0 & ~t0).sum()),
                        "P(off fixed | sur fixed, tau0 nc fail)": _stat(~f1, sf & f0, gm.log)["p"]}
    out["DAC_A3"] = _stat(d0 & ~d1, q0 & ~q1, gm.log)
    out["n_pairs"] = int(len(gm))
    if tracked:
        return out
    # official effect of guided(s) over ALL valid perturbed sources (modified or not)
    ga = g
    sa = src.loc[list(zip(ga.token, ga.k))]
    F0, F1 = (sa.nc < 1).to_numpy(), (ga.nc < 1).to_numpy()
    D0, D1 = (sa.dac < 1).to_numpy(), (ga.dac < 1).to_numpy()
    raw_clean = ~_flag(ga, "col", mc, mk)
    res = F1
    out["effect_all_sources"] = {
        "n_sources": int(len(ga)), "modified": int(ga.modified.sum()),
        "nc fail before": int(F0.sum()), "nc fixed": int((F0 & ~F1).sum()), "nc new fail": int((~F0 & F1).sum()),
        "dac fail before": int(D0.sum()), "dac fixed": int((D0 & ~D1).sum()), "dac new fail": int((~D0 & D1).sum()),
        "ttc new fail": int(((sa.ttc == 1).to_numpy() & (ga.ttc < 1).to_numpy()).sum()),
        "mean dPDMS": float((ga.pdms.to_numpy() - sa.pdms.to_numpy()).mean()),
        "mean dEP (mult=1 both)": float((ga.ep.to_numpy() - sa.ep.to_numpy())[
            (sa.mult == 1).to_numpy() & (ga.mult == 1).to_numpy()].mean()),
    }
    # LQR-tracking gap of the residual NC failures (M7b question)
    tf_s, tf_h = (ga[f"t_g_s_{mk}"] < mc).to_numpy(), (ga[f"t_g_h_{mk}"] <= 0).to_numpy()
    out["residual_nc_gap"] = {
        "residual nc failures (all sources)": int(res.sum()),
        "... raw reference clean at m_col (invisible)": int((res & raw_clean).sum()),
        "... of those, tracked states flag at m_col": int((res & raw_clean & tf_s).sum()),
        "... of those, tracked states overlap (margin 0)": int((res & raw_clean & tf_h).sum()),
        "... raw clean, source was nc fail (unfixed)": int((res & raw_clean & F0).sum()),
        "... raw clean, new failure": int((res & raw_clean & ~F0).sum()),
        "residual on modified pairs only (dev-comparable)": int((res & ga.modified.to_numpy()).sum()),
        "... raw clean (modified pairs)": int((res & raw_clean & ga.modified.to_numpy()).sum()),
        "... tracked flags them (modified pairs)": int((res & raw_clean & tf_s & ga.modified.to_numpy()).sum()),
    }
    rdac = D1
    rclean = ~_flag(ga, "dac", md, mk)
    out["residual_dac_gap"] = {
        "residual dac failures (all sources)": int(rdac.sum()),
        "... raw reference clean at m_dac": int((rdac & rclean).sum()),
        "... of those, tracked states flag at m_dac": int((rdac & rclean & (ga.t_dmin < md).to_numpy()).sum()),
    }
    return out


def a1_miss_breakdown(df: pd.DataFrame, s: dict) -> dict:
    """bank NC failures NOT flagged at setting s (raw reference): family, object type, time, raw / tracked gaps."""
    from navsim.agents.para_ssr.refiner import decoder as D
    bank = df[df.pool == "bank"]
    f = bank[bank.nc < 1]
    miss = f[~_flag(f, "col", s["m_col"], s["mask"])]
    tf = _flag(miss, "col", s["m_col"], s["mask"], tracked=True)
    th = (miss[f"t_g_h_{s['mask']}"] <= 0).to_numpy()
    q = lambda x, p: float(np.percentile(x, p)) if len(x) else float("nan")  # noqa: E731
    return {"n_nc_fail": int(len(f)), "n_miss": int(len(miss)),
            "miss by family": {D.FAMILY_NAME.get(int(k), str(k)): int(v) for k, v in miss.fam.value_counts().items()},
            "nc fail by family": {D.FAMILY_NAME.get(int(k), str(k)): int(v) for k, v in f.fam.value_counts().items()},
            "miss by nc_obj_type": {str(k): int(v) for k, v in miss.nc_obj_type.value_counts().items()},
            "miss nc_time_idx median / p10 / p90": [q(miss.nc_time_idx, 50), q(miss.nc_time_idx, 10),
                                                   q(miss.nc_time_idx, 90)],
            "miss raw min smooth g [m] median / p10 / p90": [q(miss[f"g_s_{s['mask']}"], 50),
                                                            q(miss[f"g_s_{s['mask']}"], 10),
                                                            q(miss[f"g_s_{s['mask']}"], 90)],
            "miss flagged on tracked states at m_col": int(tf.sum()),
            "miss overlapping on tracked states (margin 0)": int(th.sum()),
            "miss raw min smooth g is +inf (no counted pair at all)": int(np.isinf(miss[f"g_s_{s['mask']}"]).sum())}


def feasible(m: dict) -> dict:
    c = {"A1": m["A1"]["p"] >= A1_MIN, "A2": m["A2"]["p"] <= A2_MAX, "A3": m["A3"]["p"] >= A3_MIN,
         "DAC_human_FA": m["DAC_human_FA"]["p"] <= DAC_HFA_MAX}
    c = {k: bool(v) for k, v in c.items()}
    short = (max(0.0, A1_MIN - m["A1"]["p"]) + max(0.0, m["A2"]["p"] - A2_MAX) + max(0.0, A3_MIN - m["A3"]["p"])
             + max(0.0, m["DAC_human_FA"]["p"] - DAC_HFA_MAX))
    return {"criteria": c, "all": all(c.values()), "n_failed": int(sum(not v for v in c.values())),
            "shortfall": float(short)}


def select_setting(table: pd.DataFrame) -> dict:
    """pre-stated rule (SELECTION_RULE.txt) on a table with columns sid, m_col, m_dac, mask, A1, A2, A3, DAC_hFA,
    feasible, n_failed, shortfall."""
    t = table.copy()
    t["mask_pref"] = t["mask"].map(MASK_PREF)
    F = t[t.feasible]
    if len(F):
        F = F.sort_values(["m_col", "m_dac", "mask_pref"], ascending=False)
        return {"rule": "feasible: largest m_col, then m_dac, then current mask", "sid": int(F.sid.iloc[0]),
                "n_feasible": int(len(F)), "feasible_sids": [int(x) for x in F.sid]}
    # Pareto front over (A1 up, A2 down, A3 up, DAC_hFA down)
    v = np.stack([t.A1, -t.A2, t.A3, -t.DAC_hFA], 1)
    front = [int(t.sid.iloc[i]) for i in range(len(t))
             if not any((v[j] >= v[i]).all() and (v[j] > v[i]).any() for j in range(len(t)))]
    c = t.sort_values(["n_failed", "shortfall", "m_col", "m_dac", "mask_pref"],
                      ascending=[True, True, False, False, False])
    return {"rule": "none feasible: closest = fewest failed, then smallest shortfall, then larger margins",
            "sid": int(c.sid.iloc[0]), "n_feasible": 0, "pareto_front_sids": front}


def cmd_summarize(a):
    out = Path(a.out)
    df = pd.concat([pd.read_parquet(p) for p in sorted((out / "shards").glob("part-*.parquet"))], ignore_index=True)
    err = df[df.error != ""]
    df = df[df.error == ""].copy()
    df["sid"] = df.sid.astype(int)
    bank = df[df.pool == "bank"]
    R = {"created": time.strftime("%F %T"), "rule_file": str(REPORT_DIR / "SELECTION_RULE.txt"),
         "n_tokens": int(df.token.nunique()), "n_logs": int(df.log.nunique()), "n_rows": int(len(df)),
         "n_error_tokens": int(err.token.nunique()), "errors": err.error.head(5).tolist(),
         "bank": {"n": int(len(bank)), "n_human": int((bank.k == 0).sum()),
                  "stored-label mismatches": int((~bank.lab_ok.astype(bool)).sum()),
                  "nc fail": int((bank.nc < 1).sum()), "dac fail": int((bank.dac < 1).sum()),
                  "human nc fail": int(((bank.k == 0) & (bank.nc < 1)).sum()),
                  "human dac fail": int(((bank.k == 0) & (bank.dac < 1)).sum())},
         "grid": {"m_col": M_COLS, "m_dac": M_DACS, "mask": MASKS},
         "timing_s_per_token": {"mean": float(df.sec.dropna().mean()), "p90": float(df.sec.dropna().quantile(0.9))}}
    rows, rows_t, per = [], [], {}
    for s in SETTINGS:
        m = setting_metrics(df, s)
        f = feasible(m)
        per[s["sid"]] = {"setting": s, **m, "feasible": f}
        rows.append(dict(sid=s["sid"], mask=s["mask"], m_col=s["m_col"], m_dac=s["m_dac"],
                         A1=m["A1"]["p"], A1_lo=m["A1"]["boot_log"][0], A1_hi=m["A1"]["boot_log"][1],
                         A2=m["A2"]["p"], A2_lo=m["A2"]["boot_log"][0], A2_hi=m["A2"]["boot_log"][1],
                         A3=m["A3"]["p"], A3_lo=m["A3"]["boot_log"][0], A3_hi=m["A3"]["boot_log"][1],
                         A3_k=m["A3"]["k"], A3_n=m["A3"]["n"],
                         DAC_rec=m["DAC_recall"]["p"], DAC_rec_lo=m["DAC_recall"]["boot_log"][0],
                         DAC_rec_hi=m["DAC_recall"]["boot_log"][1], DAC_fpr=m["DAC_fpr"]["p"],
                         DAC_hFA=m["DAC_human_FA"]["p"], DAC_hFA_lo=m["DAC_human_FA"]["boot_log"][0],
                         DAC_hFA_hi=m["DAC_human_FA"]["boot_log"][1], DAC_A3=m["DAC_A3"]["p"],
                         NC_fpr=m["NC_fpr"]["p"],
                         nc_fixed=m["effect_all_sources"]["nc fixed"],
                         nc_new=m["effect_all_sources"]["nc new fail"],
                         dac_fixed=m["effect_all_sources"]["dac fixed"],
                         dac_new=m["effect_all_sources"]["dac new fail"],
                         dPDMS=m["effect_all_sources"]["mean dPDMS"],
                         res_nc=m["residual_nc_gap"]["residual nc failures (all sources)"],
                         res_nc_invisible=m["residual_nc_gap"]["... raw reference clean at m_col (invisible)"],
                         res_nc_inv_tracked=m["residual_nc_gap"]["... of those, tracked states flag at m_col"],
                         feasible=f["all"], n_failed=f["n_failed"], shortfall=f["shortfall"]))
        mt = setting_metrics(df, s, tracked=True)
        per[s["sid"]]["tracked_ceiling"] = mt
        rows_t.append(dict(sid=s["sid"], mask=s["mask"], m_col=s["m_col"], m_dac=s["m_dac"], A1_t=mt["A1"]["p"],
                           A2_t=mt["A2"]["p"], A3_t=mt["A3"]["p"], DAC_rec_t=mt["DAC_recall"]["p"],
                           DAC_hFA_t=mt["DAC_human_FA"]["p"], DAC_A3_t=mt["DAC_A3"]["p"]))
    table = pd.DataFrame(rows)
    sel = select_setting(table)
    R["selection"] = sel | {"setting": per[sel["sid"]]["setting"]}
    R["chosen"] = per[sel["sid"]]
    R["A1_miss_breakdown_chosen"] = a1_miss_breakdown(df, per[sel["sid"]]["setting"])
    R["current_default"] = per[[s["sid"] for s in SETTINGS if s["mask"] == "sat" and s["m_col"] == 0.3
                                and s["m_dac"] == 0.2][0]]
    R["per_setting"] = {str(k): v for k, v in per.items()}
    rep = Path(a.report)
    rep.mkdir(parents=True, exist_ok=True)
    table.to_csv(rep / "grid_table.csv", index=False, float_format="%.4f")
    pd.DataFrame(rows_t).to_csv(rep / "grid_tracked.csv", index=False, float_format="%.4f")
    (rep / "m8_recheck.json").write_text(json.dumps(R, indent=1, default=float))
    print(json.dumps(R["selection"], indent=1, default=float))
    print(f"-> {rep}")


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
    p.add_argument("--shard-tokens", type=int, default=10)
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--out", default=str(OUT))
    p = sub.add_parser("summarize")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--report", default=str(REPORT_DIR))
    a = ap.parse_args(argv)
    if getattr(a, "workers", 1) > 2:
        raise SystemExit("<= 2 workers (shared machine)")
    {"select": cmd_select, "run": cmd_run, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
