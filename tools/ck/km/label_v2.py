#!/usr/bin/env python
"""km: per-candidate NAVSIM v2 (EPDMS) labels of CK2's 96 submitted (laterally corrected) trajectories.

Run with the NAVSIM v2 worktree on PYTHONPATH (km_runs/v2_env.sh), NOT this repo:
  source <K>/km_runs/v2_env.sh && $PY tools/ck/km/label_v2.py --split navtrain_val --mc <v2 metric cache> \
      --root <RUN>/eval/root --run ck2e2e_bevkd --out <dir> --workers 20

Every (token, column) is scored by the official v2 ``navsim.evaluate.pdm_score.pdm_score`` (the call
run_pdm_score_from_trajectories.py makes per token: [PDM-Closed, trajectory], same simulator / scorer / traffic policy
config), so per-candidate values are the official ones by construction.  Saved per token (columns not valid -> NaN):
  lab    [96, 9]  nc dac ddc tlc ep ttc lk hc mult   (EPDMS without the two-frame term = mult * weighted / 14)
  states [96, 41, 11] float32  ego_simulated_states (for the official two-frame extended comfort between frames)
  start_time, log, frame_type
Shards are written per log (resumable); --merge assembles <out>/{lab,states,meta}.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

LAB = ("nc", "dac", "ddc", "tlc", "ep", "ttc", "lk", "hc", "mult")
COLS = ("no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
        "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound", "lane_keeping",
        "history_comfort", "multiplicative_metrics_prod")
NT = 41
_G = {}


def log(m):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {m}", flush=True)


def compose_cfg(mc: str):
    from hydra import compose, initialize_config_dir
    import navsim
    d = Path(navsim.__file__).parent / "planning/script/config/pdm_scoring"
    with initialize_config_dir(config_dir=str(d), version_base=None):
        return compose(config_name="default_run_pdm_score", overrides=[f"metric_cache_path={mc}"])


def _init(mc: str):
    from hydra.utils import instantiate
    cfg = compose_cfg(mc)
    sim = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    assert sim.proposal_sampling == scorer.proposal_sampling
    if cfg.traffic_agents == "non_reactive":
        tap = instantiate(cfg.traffic_agents_policy.non_reactive, sim.proposal_sampling)
    else:
        tap = instantiate(cfg.traffic_agents_policy.reactive, sim.proposal_sampling)
    _G.update(cfg=cfg, sim=sim, scorer=scorer, tap=tap)


def _score_log(task):
    import lzma
    import pickle
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    out_p, items = task["out"], task["items"]           # items: [(row, token, mc_path, traj[96,8,3], valid[96])]
    if Path(out_p).exists():
        return out_p, 0
    samp = TrajectorySampling(num_poses=8, interval_length=0.5)
    R = len(items)
    lab = np.full((R, 96, len(LAB)), np.nan, np.float32)
    st = np.full((R, 96, NT, 11), np.nan, np.float32)
    meta = {"row": np.zeros(R, np.int64), "start_time": np.full(R, np.nan), "frame_type": np.zeros(R, np.int64),
            "err": np.zeros(R, np.int64)}
    for r, (row, tok, mcp, traj, valid) in enumerate(items):
        meta["row"][r] = row
        try:
            with lzma.open(mcp, "rb") as f:
                mc = pickle.load(f)
            meta["start_time"][r] = mc.timepoint.time_s
            meta["frame_type"][r] = int(mc.scene_type)
            for j in np.flatnonzero(valid):
                df, ss = pdm_score(metric_cache=mc, model_trajectory=Trajectory(np.asarray(traj[j], np.float32), samp),
                                   future_sampling=_G["sim"].proposal_sampling, simulator=_G["sim"],
                                   scorer=_G["scorer"], traffic_agents_policy=_G["tap"])
                lab[r, j] = [float(df[c].iloc[0]) for c in COLS]
                st[r, j] = np.asarray(ss, np.float32)
        except Exception:
            traceback.print_exc()
            meta["err"][r] = 1
    tmp = out_p + ".tmp.npz"
    np.savez(tmp, lab=lab, states=st, **meta)
    os.replace(tmp, out_p)
    return out_p, R


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--mc", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--run", default="ck2e2e_bevkd")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N tokens")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="smallest logs first (a helper next to a running labeler)")
    a = ap.parse_args(argv)
    root, out = Path(a.root), Path(a.out)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    tok = pd.read_parquet(root / "packed" / a.split / "tokens.parquet")
    N = len(tok)
    if a.merge:
        lab = np.full((N, 96, len(LAB)), np.nan, np.float32)
        st = np.lib.format.open_memmap(out / "states.npy", "w+", np.float32, (N, 96, NT, 11))
        st[:] = np.nan
        start, ftype, err, done = np.full(N, np.nan), np.full(N, -1), np.zeros(N, np.int64), np.zeros(N, bool)
        for p in sorted((out / "shards").glob("*.npz")):
            z = np.load(p)
            r = z["row"]
            lab[r], st[r], start[r], ftype[r], err[r], done[r] = z["lab"], z["states"], z["start_time"], \
                z["frame_type"], z["err"], True
        st.flush()
        np.save(out / "lab.npy", lab)
        pd.DataFrame({"token": tok.token.astype(str), "log": tok.log.astype(str), "start_time": start,
                      "frame_type": ftype, "err": err, "done": done}).to_parquet(out / "meta.parquet")
        log(f"merged {done.sum()} / {N} tokens, errors {err.sum()}, labelled cols {np.isfinite(lab[..., 0]).sum()}")
        return 0
    lat = np.load(root / "infer" / a.run / a.split / "lat_traj.npy", mmap_mode="r")
    valid = np.load(root / "packed" / a.split / "valid96.npy", mmap_mode="r")
    mcr = Path(a.mc)
    rows = np.arange(N) if not a.limit else np.arange(a.limit)
    tasks = []
    for lg, g in tok.iloc[rows].groupby(tok.log.astype(str).iloc[rows]):
        items = []
        for row, t in zip(g.index.to_numpy(), g.token.astype(str)):
            p = mcr / lg / "unknown" / t / "metric_cache.pkl"
            if not p.exists():
                log(f"no metric cache: {t}")
                continue
            items.append((int(row), t, str(p), np.asarray(lat[row]), np.asarray(valid[row], bool)))
        if items:
            tasks.append({"out": str(out / "shards" / f"{lg}.npz"), "items": items})
    tasks.sort(key=lambda d: -len(d["items"]))
    if a.reverse:
        tasks = tasks[::-1]
    log(f"{a.split}: {sum(len(t['items']) for t in tasks)} tokens in {len(tasks)} logs, workers {a.workers}")
    import multiprocessing as mp
    t0, n = time.time(), 0
    with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(a.mc,)) as pool:
        for i, (p, r) in enumerate(pool.imap_unordered(_score_log, tasks)):
            n += r
            if i % 10 == 0:
                log(f"{i + 1}/{len(tasks)} logs, {n} tokens, {n / max(time.time() - t0, 1):.2f} tok/s")
    log(f"done {n} tokens in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
