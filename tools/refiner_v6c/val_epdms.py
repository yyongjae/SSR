#!/usr/bin/env python
"""Official NAVSIM v2.2 one-stage EPDMS (with the two-frame extended comfort EC) of refiner v6c on navtrain_val or
navtest (--split, table SPLITS).

The scorer path is the one of report 76 / 119 (read-only reuse): NAVSIM v2.2 at /workspace/navsim_workspace/navsim
(@0a380a9, post-fix pdm_score.py), non-reactive, `run_pdm_score_one_stage.py` with the precomputed agent of
/workspace/yongjae/perception_eval/epdms (traj_pkl = {token: float32 [8, 3] x, y, heading in the NAVSIM ego frame at t0,
0.5 s steps}), the val metric cache A_epdms/metric_cache_v22_val_full (18,179 / 18,179 tokens) and the quoted split
navtrain_val_e2e_q (= the 18,179 tokens of resources/km_corridor/packed/navtrain_val; checked).  Nothing is written
under A_epdms/ or perception_eval/: outputs, logs, hydra runs and subset split configs go to --out.

Submissions (the four outputs of the 10-27 feedback; report 126 §9):
  base  tau_base of the dump (v6_base.npy f32 = the cand_official pick msel; for a stage-D model this is the jointly
        trained baseline's own pick)
  v6    tau_v6 always (v6_traj.npy)
  out   the submitted trajectory (v6_switch ? tau_v6 : tau_base, the forward's R-J switch)
  c0    optional: the C0 model's own pick from another dump (--c0-dump: traj.npy[msel], the select_npz rule 'model'),
        or any select_npz output (--npz name=path.npz, keys tokens / trajectory)
EC compares the current frame's simulated plan with the previous adjacent frame's (same log, start-time gap <= 0.55 s
among the evaluated tokens, run_pdm_score_one_stage.infer_start_adjacent_mapping); frames without one get EC weight 0.

navtest (A49): the same scorer call as km's corr30_cf_v15_fix/eval_ep5/run_eval_ep5.sh (its navtest EPDMS 86.81 is
comparable): metric cache perception_eval/epdms/metric_cache_v22_navtest, the stock split navtest (data_split test,
navsim_logs/test/test, no A_epdms config on the search path) and the start times of A_epdms/sel/adjacent_navtest.parquet
(all 12,146 tokens); a subset gets a quoted split with data_split test.  The split is written into pkl_info.json by
`pkl` and read back by `run` / `analyze`; navtest outputs are named epdms_navtest_* (navtrain_val keeps epdms_val_*,
so its old csvs are reused).  tools/corridor/score_npz.py builds its scorer call with scorer_cmd().

  # 1. pickles from a dump of every navtrain_val token (dump_corr_model.py --v6 --cand --no-fields --traj32)
  $PY tools/refiner_v6c/val_epdms.py pkl --dump <dump> --out <out> [--c0-dump <c0 dump>]
  # 2. official scorer, one run per submission (CPU, ray, nice); a dump with fewer tokens gets its own quoted split
  $PY tools/refiner_v6c/val_epdms.py run --out <out> [--subs base,v6,out,c0] [--threads 16]
  # 3. paired analysis -> <out>/epdms_summary.json
  $PY tools/refiner_v6c/val_epdms.py analyze --out <out>
  # or all three
  $PY tools/refiner_v6c/val_epdms.py all --dump <dump> --out <out> [--c0-dump ...] [--threads 16]
  # navtest (eval.sh, a v6c run): the submission out + base / v6 + the same model's v2 (select_npz rule v2)
  $PY tools/refiner_v6c/val_epdms.py all --split navtest --dump <navtest dump> --out <out> --npz v2=<sel_v2.npz>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
NAVSIM22 = Path("/workspace/navsim_workspace/navsim")
NAVSIM_DATA = Path("/workspace/navsim_workspace/dataset")
PE_EPDMS = Path("/workspace/yongjae/perception_eval/epdms")                     # precomputed_agent.py + its agent config
A_EPDMS = Path("/home/external-user/ssd/yongjae_refiner/ck/ck2/analysis/ep_noim_check/A_epdms")
METRIC_CACHE = A_EPDMS / "metric_cache_v22_val_full"
FULL_SPLIT = "navtrain_val_e2e_q"                                               # A_epdms/config (all 18,179 tokens)
ADJ_FULL = A_EPDMS / "sel" / "adjacent_navtrain_val.parquet"                    # token, log_name, t (s), prev
MET = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance",
       "ego_progress", "time_to_collision_within_bound", "lane_keeping", "history_comfort", "two_frame_extended_comfort"]
SHORT = dict(zip(MET, ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]))
PAIRS = (("out", "base"), ("v6", "base"), ("out", "c0"), ("base", "c0"), ("v6", "c0"))
# per evaluation split (A49): metric cache, full split (navtrain_val: the quoted FULL_SPLIT; navtest: the stock split),
# navsim logs, adjacency table (its tokens = the full split), scene data_split of a subset, extra hydra search path
SPLITS = {
    "navtrain_val": dict(metric_cache=METRIC_CACHE, full_split=FULL_SPLIT,
                         logs=NAVSIM_DATA / "navsim_logs/trainval/trainval", adj=ADJ_FULL, data_split="trainval",
                         searchpath=[f"file://{A_EPDMS}/config"]),
    "navtest": dict(metric_cache=PE_EPDMS / "metric_cache_v22_navtest", full_split="navtest",
                    logs=NAVSIM_DATA / "navsim_logs/test/test", adj=A_EPDMS / "sel" / "adjacent_navtest.parquet",
                    data_split="test", searchpath=[]),
}


def kst() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S KST", time.gmtime(time.time() + 9 * 3600))


# ------------------------------------------------------------------------------------------------------ helpers
def table(split: str) -> dict:
    """SPLITS[split]; the navtrain_val paths are read from the module constants at call time (tests patch ADJ_FULL)"""
    if split not in SPLITS:
        raise KeyError(f"no EPDMS table for split {split!r}; known {sorted(SPLITS)}")
    t = dict(SPLITS[split])
    if split == "navtrain_val":
        t.update(metric_cache=METRIC_CACHE, full_split=FULL_SPLIT, adj=ADJ_FULL)
    return t


def prefix(split: str) -> str:
    """output name prefix: epdms_val for navtrain_val (unchanged, old csvs are reused), epdms_<split> otherwise"""
    return "epdms_val" if split == "navtrain_val" else f"epdms_{split}"


def log_bootstrap(diff: np.ndarray, logs: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Paired log-cluster bootstrap of mean(diff) (= tools/ck/eval_ck.log_bootstrap, copied to avoid its imports):
    resample logs with replacement, ratio estimator over tokens; NaN rows dropped."""
    diff = np.asarray(diff, np.float64)
    logs = np.asarray(logs)
    m = np.isfinite(diff)
    diff, logs = diff[m], logs[m]
    if len(diff) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "n_logs": 0}
    ul, inv = np.unique(logs, return_inverse=True)
    L = len(ul)
    S = np.bincount(inv, weights=diff, minlength=L)
    C = np.bincount(inv, minlength=L).astype(np.float64)
    W = np.random.default_rng(seed).multinomial(L, np.full(L, 1.0 / L), size=n_boot).astype(np.float64)
    est = (W @ S) / np.maximum(W @ C, 1e-12)
    lo, hi = np.quantile(est, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(diff.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(diff)), "n_logs": int(L)}


def adjacency(tokens: List[str], adj_full: Optional[Path] = None) -> pd.DataFrame:
    """previous adjacent token among `tokens` (official infer_start_adjacent_mapping: same log, sorted by start time,
    consecutive gap <= 0.55 s), from the start times of the report-76 adjacency table (all 18,179 val tokens; navtest:
    pass SPLITS['navtest']['adj'])."""
    adj_full = Path(adj_full or ADJ_FULL)
    full = pd.read_parquet(adj_full)[["token", "log_name", "t"]]
    full["token"] = full.token.astype(str)
    miss = set(tokens) - set(full.token)
    assert not miss, f"{len(miss)} tokens have no start time in {adj_full} (e.g. {sorted(miss)[:3]})"
    df = full[full.token.isin(set(tokens))].sort_values(["log_name", "t"]).reset_index(drop=True)
    prev: List[Optional[str]] = [None] * len(df)
    for i in range(1, len(df)):
        if df.log_name[i] == df.log_name[i - 1] and abs(df.t[i] - df.t[i - 1]) <= 0.55:
            prev[i] = df.token[i - 1]
    df["prev"] = prev
    return df


def recompose(df: pd.DataFrame) -> np.ndarray:
    """v2.2 one-stage score from the csv sub-metrics: mult x (5 EP + 5 TTC + 2 LK + 2 HC [+ 2 EC]) / (14 [+ 2])."""
    mult = df[MET[:4]].prod(axis=1).to_numpy()
    ec = df[MET[8]].to_numpy(np.float64)
    num = (df[MET[4:8]].to_numpy() * np.array([5, 5, 2, 2])).sum(1) + np.where(np.isnan(ec), 0.0, 2 * np.nan_to_num(ec))
    return mult * num / (14 + np.where(np.isnan(ec), 0, 2))


def no_ec(df: pd.DataFrame) -> np.ndarray:
    """counterfactual score with the EC term dropped on every row."""
    return df[MET[:4]].prod(axis=1).to_numpy() * (df[MET[4:8]].to_numpy() * np.array([5, 5, 2, 2])).sum(1) / 14


def dump_rows(d: Path) -> np.ndarray:
    """rows written by dump_corr_model.py (union of its done_<worker>.npy masks)"""
    ref = next(d / f for f in ("v6_base.npy", "msel.npy", "traj.npy") if (d / f).exists())
    done = np.zeros(len(np.load(ref, mmap_mode="r")), bool)
    for f in d.glob("done_*.npy"):
        done |= np.load(f)
    return np.flatnonzero(done)


def check_traj(name: str, T: Dict[str, np.ndarray]):
    for t, x in T.items():
        assert x.shape == (8, 3) and x.dtype == np.float32 and np.isfinite(x).all(), (name, t, x.shape, x.dtype)


# ------------------------------------------------------------------------------------------------------ 1. pkl
def cmd_pkl(a) -> int:
    out = Path(a.out)
    (out / "traj").mkdir(parents=True, exist_ok=True)
    root = os.environ.get("KM_CORR_ROOT", str(REPO / "resources/km_corridor"))
    tdf = pd.read_parquet(f"{root}/packed/{a.split}/tokens.parquet")
    toks_all, logs_all = tdf.token.astype(str).to_numpy(), tdf.log.astype(str).to_numpy()
    d = Path(a.dump) / a.split
    rows = dump_rows(d)
    assert len(rows), f"no dumped rows in {d}"
    base = np.asarray(np.load(d / "v6_base.npy", mmap_mode="r")[rows], np.float32)
    v6 = np.asarray(np.load(d / "v6_traj.npy", mmap_mode="r")[rows], np.float32)
    sw = np.asarray(np.load(d / "v6_switch.npy", mmap_mode="r")[rows]).astype(bool)
    outt = np.where(sw[:, None, None], v6, base)
    toks = toks_all[rows]
    subs = {"base": base, "v6": v6, "out": outt}
    if a.c0_dump:                                      # the C0 model's own pick (select_npz rule 'model')
        dc = Path(a.c0_dump) / a.split
        rc = dump_rows(dc)
        assert set(rows) <= set(rc), f"C0 dump {dc} lacks {len(set(rows) - set(rc))} of the v6c dump's rows"
        msel = np.asarray(np.load(dc / "msel.npy", mmap_mode="r")[rows], np.int64)
        tr = np.load(dc / "traj.npy", mmap_mode="r")
        if tr.dtype != np.float32:
            print(f"WARNING: {dc}/traj.npy is {tr.dtype} (dump without --traj32): ~cm rounding of the C0 plan", flush=True)
        subs["c0"] = np.stack([np.asarray(tr[r], np.float32)[c] for r, c in zip(rows, msel)])
    for spec in a.npz or []:                           # name=path from select_npz.py
        name, p = spec.split("=", 1)
        z = np.load(p, allow_pickle=True)                 # select_npz.py stores tokens as an object array
        m = dict(zip(z["tokens"].astype(str), np.asarray(z["trajectory"], np.float32)))
        miss = [t for t in toks if t not in m]
        assert not miss, f"{p} lacks {len(miss)} tokens"
        subs[name] = np.stack([m[t] for t in toks])
    info = {"created": kst(), "dump": str(d), "split": a.split, "n_tokens": int(len(rows)),
            "switch_rate": float(sw.mean()), "subs": {}}
    for name, X in subs.items():
        T = {t: np.ascontiguousarray(x, np.float32) for t, x in zip(toks, X)}
        check_traj(name, T)
        assert len(T) == len(rows), f"duplicate tokens in {name}"
        p = out / "traj" / f"{name}.pkl"
        with open(p, "wb") as f:
            pickle.dump(T, f)
        info["subs"][name] = {"pkl": str(p), "md5": hashlib.md5(p.read_bytes()).hexdigest()}
    pd.DataFrame({"token": toks, "log": logs_all[rows], "row": rows, "switch": sw}).to_parquet(out / "tokens.parquet")
    dmax = np.abs(v6 - base).max(axis=(1, 2))
    info["v6_minus_base_max_abs"] = {"max": float(dmax.max()), "median": float(np.median(dmax))}
    json.dump(info, open(out / "pkl_info.json", "w"), indent=1)
    print(json.dumps(info, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------------ 2. run
def split_for(out: Path, toks: List[str], logs: List[str], split: str = "navtrain_val") -> str:
    """the full split of `split` (table SPLITS) when the dump covers every token of its adjacency table, else a quoted
    subset split under <out>/config with the table's data_split (tokens and log names single-quoted: OmegaConf parses
    tokens like 3315880386e45927 as floats otherwise)."""
    T = table(split)
    full = pd.read_parquet(T["adj"]).token.astype(str)                           # navtrain_val: 18,179 = FULL_SPLIT
    if set(toks) == set(full):
        return T["full_split"]
    from omegaconf import OmegaConf
    h = hashlib.md5("\n".join(sorted(toks)).encode()).hexdigest()[:8]
    tag = "val" if split == "navtrain_val" else split
    name = f"v6c_{tag}_{len(toks)}_{h}_q"
    C = out / "config" / "train_test_split"
    (C / "scene_filter").mkdir(parents=True, exist_ok=True)
    ul = sorted(set(logs))
    lines = [f"# refiner v6c {tag} EPDMS subset ({len(toks)} tokens, {len(ul)} logs), quoted",
             "_target_: navsim.common.dataclasses.SceneFilter", "_convert_: 'all'",
             "num_history_frames: 4", "num_future_frames: 10", "frame_interval: 1", "has_route: true", "max_scenes: null",
             "log_names:"] + [f"  - '{x}'" for x in ul] + ["tokens:"] + [f"  - '{x}'" for x in toks]
    p = C / "scene_filter" / f"{name}.yaml"
    p.write_text("\n".join(lines) + "\n")
    (C / f"{name}.yaml").write_text(f"defaults:\n  - scene_filter: {name}\n\ndata_split: {T['data_split']}\n")
    cfg = OmegaConf.load(p)
    assert list(cfg.tokens) == list(toks) and all(isinstance(t, str) for t in cfg.tokens), "token round-trip failed"
    assert list(cfg.log_names) == ul, "log round-trip failed"
    return name


def current_subs(out: Path) -> Dict[str, str]:
    """{submission: pickle md5} written by the LAST `pkl` call (older pickles left in traj/ are ignored)"""
    info = json.load(open(out / "pkl_info.json"))
    return {k: v["md5"] for k, v in info["subs"].items()}


def split_of(out: Path) -> str:
    """the split of the LAST `pkl` call (pkl_info.json; navtrain_val when absent: outputs older than SPLITS)"""
    p = out / "pkl_info.json"
    return str(json.load(open(p)).get("split", "navtrain_val")) if p.exists() else "navtrain_val"


def run_dir(out: Path, sub: str, md5: str, split: str = "navtrain_val") -> Path:
    """scorer output dir of one submission, keyed by its pickle's md5: a csv is reused only for the same pickle"""
    return out / "exp" / f"{prefix(split)}_{sub}_{md5[:12]}"


def csv_of(out: Path, sub: str) -> Optional[Path]:
    """the newest scorer csv of the CURRENT pickle of `sub` (a --force re-run writes a new dated dir next to it)"""
    md5 = current_subs(out).get(sub)
    if md5 is None:
        return None
    base = run_dir(out, sub, md5, split_of(out))
    c = [x for d in sorted(base.parent.glob(base.name + "*")) for x in d.rglob("*.csv")]
    return max(c, key=lambda x: x.stat().st_mtime) if c else None


def scorer_cmd(split: str, pkl, exp_root, exp_name: str, output_dir, threads: int, nice: int,
               extra_searchpath: Sequence[str] = (), tts: Optional[str] = None) -> Tuple[List[str], dict, Path]:
    """(cmd, env, cwd) of one official v2.2 one-stage scorer run of a {token: f32 [8, 3]} pickle on SPLITS[split]
    (precomputed agent, ray); tts = the train_test_split to score (default: the split's full one; a subset adds its
    <out>/config through extra_searchpath).  navtest = km's run_eval_ep5.sh call."""
    T = table(split)
    env = {**os.environ, "NAVSIM_DEVKIT_ROOT": str(NAVSIM22), "OPENSCENE_DATA_ROOT": str(NAVSIM_DATA),
           "NUPLAN_MAPS_ROOT": str(NAVSIM_DATA / "maps"), "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0",
           "NAVSIM_EXP_ROOT": str(exp_root), "PYTHONPATH": f"{NAVSIM22}:{PE_EPDMS}", "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    sp = ["pkg://navsim.planning.script.config.common", f"file://{PE_EPDMS}/config"] + list(T["searchpath"]) \
        + list(extra_searchpath)
    cmd = ["nice", "-n", str(nice), sys.executable, "navsim/planning/script/run_pdm_score_one_stage.py",
           f"train_test_split={tts or T['full_split']}", f"navsim_log_path={T['logs']}",
           f"metric_cache_path={T['metric_cache']}", "agent=precomputed_agent", f"agent.traj_pkl={pkl}",
           f"experiment_name={exp_name}", f"output_dir={output_dir}",
           "worker=ray_distributed", f"worker.threads_per_node={threads}",
           "hydra.searchpath=[" + ",".join(sp) + "]"]
    return cmd, env, NAVSIM22


def cmd_run(a) -> int:
    out = Path(a.out).resolve()
    tk = pd.read_parquet(out / "tokens.parquet")
    sp_name = split_of(out)
    split = split_for(out, tk.token.astype(str).tolist(), tk.log.astype(str).tolist(), sp_name)
    cur = current_subs(out)
    subs = [s for s in a.subs.split(",") if s in cur]
    (out / "logs").mkdir(exist_ok=True)
    extra = [] if split == table(sp_name)["full_split"] else [f"file://{out}/config"]
    pre = prefix(sp_name)
    rec = {"split": split, "eval_split": sp_name, "runs": {}}
    for s in subs:
        old = csv_of(out, s)
        if old is not None and not a.force:
            print(f"[{kst()}] skip {s}: csv of this pickle exists ({old})", flush=True)
            continue
        rd = run_dir(out, s, cur[s], sp_name)
        if old is not None:                                   # --force: a new dated dir, the old csv is kept
            rd = rd.with_name(rd.name + "_" + time.strftime("%Y%m%d%H%M%S"))
        cmd, env, cwd = scorer_cmd(sp_name, f"{out}/traj/{s}.pkl", out / "exp", f"{pre}_v6c_{s}", rd, a.threads,
                                   a.nice, extra, tts=split)
        t0 = time.time()
        print(f"[{kst()}] start {s} ({split}, {a.threads} threads)", flush=True)
        with open(out / "logs" / f"{pre}_{s}.log", "w") as lf:
            rc = subprocess.run(cmd, cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        c = csv_of(out, s)
        rec["runs"][s] = {"rc": rc, "csv": str(c) if c else None, "sec": round(time.time() - t0, 1), "end": kst(),
                          "cmd": " ".join(cmd)}
        print(f"[{kst()}] end {s} rc={rc} csv={c}", flush=True)
        if rc != 0 or c is None:
            json.dump(rec, open(out / "run_info.json", "w"), indent=1)
            return 1
    json.dump(rec, open(out / "run_info.json", "w"), indent=1)
    return 0


# ------------------------------------------------------------------------------------------------------ 3. analyze
def load_csv(p: Path):
    df = pd.read_csv(p, index_col=0, dtype={"token": str})
    avg = df[df.token == "average_all_frames"].iloc[0]
    df = df[df.token != "average_all_frames"].copy()
    df["valid"] = df["valid"].astype(str) == "True"
    for c in MET + ["score"]:
        df[c] = df[c].astype(float)
    return df.set_index("token"), avg


def analyze(out: Path, n_boot: int = 2000) -> dict:
    tk = pd.read_parquet(out / "tokens.parquet")
    toks = tk.token.astype(str).tolist()
    logs = tk.log.astype(str).to_numpy()
    sw = tk.switch.to_numpy().astype(bool)
    D, P = {}, {}
    cur = current_subs(out)
    for s in ("base", "v6", "out", "c0") + tuple(sorted(cur)):
        if s in D or s not in cur or csv_of(out, s) is None:
            continue
        D[s] = load_csv(csv_of(out, s))
        P[s] = pickle.load(open(out / "traj" / f"{s}.pkl", "rb"))
    assert "base" in D, "no base csv: run first"
    adj = adjacency(toks, table(split_of(out))["adj"]).set_index("token").loc[toks]
    has_prev = adj.prev.notna().to_numpy()
    pos = {t: i for i, t in enumerate(toks)}
    prev_idx = np.array([pos[p] if isinstance(p, str) else -1 for p in adj.prev])
    res = {"created": kst(), "out": str(out), "split": split_of(out), "n_tokens": len(toks),
           "n_logs": int(len(np.unique(logs))),
           "n_with_prev": int(has_prev.sum()), "switch_rate": float(sw.mean()), "abs": {}, "checks": {},
           "paired": {}, "change": {}, "ddc_flips": {}}
    for s, (df, avg) in D.items():
        assert sorted(df.index) == sorted(toks), f"{s}: csv token set differs from the dump"
        df = df.loc[toks]
        D[s] = (df, avg)
        res["checks"][f"{s}_all_valid"] = bool(df.valid.all())
        ec_nan = df[MET[8]].isna().to_numpy()
        res["checks"][f"{s}_ec_nan_equals_no_prev"] = bool((ec_nan == ~has_prev).all())
        res["checks"][f"{s}_score_recompose_maxabs"] = float(np.abs(recompose(df) - df.score.to_numpy()).max())
        res["abs"][s] = {**{SHORT[m]: float(100 * np.nanmean(df[m])) for m in MET},
                         "EPDMS": float(100 * df.score.mean()), "EPDMS_noEC": float(100 * no_ec(df).mean()),
                         "EPDMS_csv_avg_row": float(100 * float(avg["score"])), "n_EC": int((~ec_nan).sum())}
    for A, B in PAIRS:
        if A not in D or B not in D:
            continue
        a_, b_ = D[A][0], D[B][0]
        key = f"{A}_minus_{B}"
        pr = {"EPDMS": log_bootstrap(100 * (a_.score - b_.score).to_numpy(), logs, n_boot),
              "EPDMS_noEC": log_bootstrap(100 * (no_ec(a_) - no_ec(b_)), logs, n_boot)}
        for m in MET:
            pr[SHORT[m]] = log_bootstrap(100 * (a_[m] - b_[m]).to_numpy(), logs, n_boot)
        res["paired"][key] = pr
        ta = np.stack([P[A][t] for t in toks])
        tb = np.stack([P[B][t] for t in toks])
        tr_ch = np.abs(ta - tb).max(axis=(1, 2)) > 1e-4
        prev_ch = np.where(prev_idx >= 0, tr_ch[np.maximum(prev_idx, 0)], False)
        pair_ch = tr_ch | prev_ch
        ea, eb = a_[MET[8]].to_numpy(), b_[MET[8]].to_numpy()
        ch = {"traj_changed_pct": float(100 * tr_ch.mean()),
              "endpoint_l2_mean_m": float(np.linalg.norm(ta[:, -1, :2] - tb[:, -1, :2], axis=1).mean()),
              "ec_identical_on_unchanged_pairs": bool(np.array_equal(ea[has_prev & ~pair_ch], eb[has_prev & ~pair_ch]))}
        for nm, msk in (("cur_changed", has_prev & tr_ch), ("cur_unchanged", has_prev & ~tr_ch),
                        ("pair_changed", has_prev & pair_ch), ("pair_unchanged", has_prev & ~pair_ch)):
            ch[nm] = {"n": int(msk.sum()),
                      f"EC_{A}": float(100 * ea[msk].mean()) if msk.any() else None,
                      f"EC_{B}": float(100 * eb[msk].mean()) if msk.any() else None}
        res["change"][key] = ch
        da, db = a_[MET[2]].to_numpy(), b_[MET[2]].to_numpy()
        res["ddc_flips"][key] = {"n_differ": int((da != db).sum()), f"{A}_lower": int((da < db).sum()),
                                 f"{A}_higher": int((da > db).sum()), "pct_differ": float(100 * (da != db).mean())}
    json.dump(res, open(out / "epdms_summary.json", "w"), indent=1)
    return res


def cmd_analyze(a) -> int:
    res = analyze(Path(a.out).resolve(), a.n_boot)
    print(json.dumps({k: res[k] for k in ("n_tokens", "n_with_prev", "switch_rate", "abs", "checks", "ddc_flips")}, indent=1))
    for k, pr in res["paired"].items():
        e = pr["EPDMS"]
        print(f"{k}: EPDMS {e['mean']:+.3f} [{e['lo']:+.3f}, {e['hi']:+.3f}]  EC {pr['EC']['mean']:+.3f}  "
              f"DDC {pr['DDC']['mean']:+.3f}  traj changed {res['change'][k]['traj_changed_pct']:.1f} %")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("pkl", "run", "analyze", "all"):
        p = sub.add_parser(nm)
        p.add_argument("--out", required=True)
        if nm in ("pkl", "all"):
            p.add_argument("--dump", required=True)
            p.add_argument("--split", default="navtrain_val")
            p.add_argument("--c0-dump", default="")
            p.add_argument("--npz", action="append", help="name=path.npz (select_npz.py output), extra submission")
        if nm in ("run", "all"):
            p.add_argument("--subs", default="base,v6,out,c0")
            p.add_argument("--threads", type=int, default=16)
            p.add_argument("--nice", type=int, default=10)
            p.add_argument("--force", action="store_true")
        if nm in ("analyze", "all"):
            p.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    if a.cmd == "pkl":
        return cmd_pkl(a)
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "analyze":
        return cmd_analyze(a)
    if a.cmd == "all":
        if a.npz:
            names = ",".join(s.split("=", 1)[0] for s in a.npz)
            a.subs = a.subs + "," + names
        return cmd_pkl(a) or cmd_run(a) or cmd_analyze(a)
    return 2


if __name__ == "__main__":
    sys.exit(main())
