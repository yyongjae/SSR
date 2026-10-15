#!/usr/bin/env python
"""navsim v2 one-stage EPDMS of stage-T refiner drafts on navtest (PRESTATED_DECISION_RULE AMENDMENT 6 addendum;
reported, NOT decisive).  CPU only.

  # throughput measurement (50 tokens x 13 original drafts; nothing is written unless --out is given)
  python tools/refiner/epdms_navtest.py bench --runs <runs>/stageT4_T_fold0_seed0 --n-tokens 50 --workers 4
  # scoring: original drafts (once, arm-independent) + refined drafts tau1 of every run, same tokens
  python tools/refiner/epdms_navtest.py score --runs <run_T> <run_M> <run_TM> <run_none> --eval eval_navtest \
      --n-tokens <n> --subset-seed 0 --workers 8 --out <data>/epdms/stageT4_navtest
  # EPDMS of the FINAL trajectories (tau1 where p_g >= theta_arm, else the original), per arm and per pair
  python tools/refiner/epdms_navtest.py report --scores <data>/epdms/stageT4_navtest \
      --selection report/refiner_T/run4/selection.json --pairs T:none,M:none,TM:T,TM:M --out <json>

Scorer (report/29_navsim_metric_versions.md): the user's navsim_v2 worktree /home/external-user/yongjae/navsim_v2
(HEAD 0a380a9 = v2.2 tag + README; includes the Issue #151 human-filter fix), its navtest metric cache
/home/external-user/yongjae/navsim_v2_exp/exp/metric_cache_navtest, config pdm_scoring/default_run_pdm_score
(non-reactive traffic agents, PDMScorer weights EP 5 / TTC 5 / LK 2 / HC 2 / EC 2, multiplicative NC, DAC, DDC, TLC,
human_penalty_filter True).  Every draft is scored by the UNMODIFIED navsim.evaluate.pdm_score.pdm_score (simulation,
scoring, human filter) and composed by the unmodified run_pdm_score_one_stage.compute_final_scores.
The 'score' / 'bench' sub-commands re-execute this file with the v2 environment (the para_ssr_env.sh variables; the
SSR repository's own 'navsim' package must NOT be importable there) -- navsim.__file__ is checked.  'report' runs in
the SSR environment (no scorer).

Deviation from the leaderboard EPDMS (stated): the two-frame extended comfort (EC) needs ONE planner's trajectories on
adjacent frames; a draft bank has 13 unrelated perturbations per token (and a token subset has almost no adjacent
pairs), so EC is not computed: 'score' = compute_final_scores with two_frame_extended_comfort = NaN, which is exactly
the official treatment of a frame without a previous adjacent frame (EC weight 0).  Applied identically to originals
and to every arm.

Drafts: pred.npz of <run>/<eval> (eval_refiner.py predict): tau0 [N, 13, 8, 3] (the bank drafts, identical bytes in
every run -- checked), tau1 (decoded correction, ungated), p_g, draft_valid, family.  Invalid drafts are not scored.
A tau1 whose bytes equal tau0 is not re-scored: it gets the original's row (flag copied_from_orig; the scorer is a
deterministic function of the trajectory and the token).
Token subset (--n-tokens n, --subset-seed s): sorted(unique tokens of the eval) -> numpy default_rng(s).choice(n, no
replacement), sorted; all 13 drafts of each chosen token; the list and its sha256 are written to meta.json.  Without
--n-tokens every token of the eval is scored.
Outputs (--out DIR): rows.parquet (set = 'orig' | run name, arm, token, k, family, valid, p_g, copied_from_orig,
the v2 metric columns, score, error), meta.json, shards/ (resumable per chunk of tokens).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

NAV = Path("/home/external-user/yongjae/navsim_v2")
V2_EXP = Path("/home/external-user/yongjae/navsim_v2_exp")
MC_NAVTEST = V2_EXP / "exp" / "metric_cache_navtest"
PYTHON = "/home/external-user/miniconda3/envs/ssr/bin/python"
METRIC_COLS = ("no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
               "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound", "lane_keeping",
               "history_comfort", "multiplicative_metrics_prod")
MAX_WORKERS = 8
V2_FLAG = "EPDMS_V2_ENV"


# ----------------------------------------------------------------------------------------------- environment
def v2_env(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The para_ssr_env.sh environment (v2 worktree first on PYTHONPATH), CPU only, 1 thread per process."""
    e = dict(os.environ if base is None else base)
    e.update(PYTHONPATH=str(NAV), NAVSIM_DEVKIT_ROOT=str(NAV), OPENSCENE_DATA_ROOT=str(V2_EXP / "dataset"),
             NAVSIM_EXP_ROOT=str(V2_EXP / "exp"), NUPLAN_MAP_VERSION="nuplan-maps-v1.0",
             NUPLAN_MAPS_ROOT=str(V2_EXP / "dataset" / "maps"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
             CUDA_VISIBLE_DEVICES="")
    e[V2_FLAG] = "1"
    return e


def ensure_v2(argv: Sequence[str]) -> None:
    """Re-exec this script under the v2 environment unless already there; then check which navsim is imported."""
    if os.environ.get(V2_FLAG) != "1":
        os.chdir("/tmp")                               # never let the SSR working directory shadow the v2 package
        os.execve(PYTHON, [PYTHON, str(Path(__file__).resolve()), *argv], v2_env())
    import navsim

    if not str(Path(navsim.__file__).resolve()).startswith(str(NAV)):
        raise SystemExit(f"navsim resolves to {navsim.__file__}, expected the v2 worktree {NAV}")


# ----------------------------------------------------------------------------------------------- inputs
def pick_tokens(tokens: Sequence[str], n: Optional[int], seed: int = 0) -> List[str]:
    """Pre-stated random token subset: sorted unique tokens -> default_rng(seed).choice(n, replace=False), sorted.
    n None (or >= all) -> every token (sorted)."""
    u = sorted(set(map(str, tokens)))
    if n is None or n >= len(u):
        return u
    idx = np.random.default_rng(int(seed)).choice(len(u), int(n), replace=False)
    return sorted(u[i] for i in idx)


def token_hash(tokens: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(tokens).encode()).hexdigest()


def run_arm(run: Path) -> str:
    return json.loads((Path(run) / "config.json").read_text())["arm"]


def load_sets(runs: Sequence[Path], eval_name: str, tokens: Optional[Sequence[str]] = None, orig: bool = True,
              refined: bool = True) -> Dict:
    """-> dict(tokens [n], family [n, K], valid [n, K], sets = {name: dict(arm, traj [n, K, 8, 3] f32, p_g [n, K],
    copy [n, K] bool)}).  'orig' = tau0 (identical in every run -- asserted), one set per run = its tau1."""
    preds = {}
    for r in runs:
        with np.load(Path(r) / eval_name / "pred.npz", allow_pickle=False) as P:
            preds[Path(r).name] = (Path(r), {k: P[k] for k in ("tokens", "tau0", "tau1", "p_g", "draft_valid", "family")})
    names = list(preds)
    ref = preds[names[0]][1]
    for n in names[1:]:
        P = preds[n][1]
        if not (np.array_equal(P["tokens"], ref["tokens"]) and np.array_equal(P["tau0"], ref["tau0"])
                and np.array_equal(P["draft_valid"], ref["draft_valid"])):
            raise ValueError(f"{n}: tokens / tau0 / draft_valid differ from {names[0]} (not the same draft bank)")
    pos = {str(t): i for i, t in enumerate(ref["tokens"])}
    tokens = sorted(pos) if tokens is None else list(tokens)
    miss = [t for t in tokens if t not in pos]
    if miss:
        raise ValueError(f"{len(miss)} requested tokens are not in the eval (e.g. {miss[:3]})")
    ix = np.array([pos[t] for t in tokens], np.int64)
    out = dict(tokens=tokens, family=ref["family"][ix], valid=ref["draft_valid"][ix].astype(bool), sets={})
    tau0 = ref["tau0"][ix].astype(np.float32)
    if orig:
        out["sets"]["orig"] = dict(arm="orig", traj=tau0, p_g=np.full(tau0.shape[:2], np.nan),
                                   copy=np.zeros(tau0.shape[:2], bool))
    if refined:
        for n in names:
            run, P = preds[n]
            t1 = P["tau1"][ix].astype(np.float32)
            out["sets"][n] = dict(arm=run_arm(run), traj=t1, p_g=P["p_g"][ix].astype(np.float64),
                                  copy=(t1 == tau0).all((2, 3)))
    return out


# ----------------------------------------------------------------------------------------------- scoring (v2 env)
_W: Dict = {}


def compose_cfg():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(NAV / "navsim/planning/script/config/pdm_scoring"), version_base=None):
        return compose(config_name="default_run_pdm_score",
                       overrides=["train_test_split=navtest", "experiment_name=epdms_refiner",
                                  f"metric_cache_path={MC_NAVTEST}"])


def _worker_init(cfg=None):
    from hydra.utils import instantiate
    from navsim.common.dataloader import MetricCacheLoader
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    cfg = cfg if cfg is not None else compose_cfg()
    assert cfg.traffic_agents == "non_reactive" and bool(cfg.scorer.config.human_penalty_filter)
    sim = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    assert sim.proposal_sampling == scorer.proposal_sampling
    _W.update(sim=sim, scorer=scorer, policy=instantiate(cfg.traffic_agents_policy.non_reactive, sim.proposal_sampling),
              loader=MetricCacheLoader(Path(cfg.metric_cache_path)),
              sampling=TrajectorySampling(num_poses=8, interval_length=0.5))


def final_score_noec(row: pd.DataFrame) -> float:
    """Official composition (run_pdm_score_one_stage.compute_final_scores) with two-frame EC = NaN (weight 0)."""
    from navsim.planning.script.run_pdm_score_one_stage import compute_final_scores

    df = row.copy()
    df["two_frame_extended_comfort"] = np.nan
    return float(compute_final_scores(df)["score"].iloc[0])


def score_token(task) -> List[Dict]:
    """Worker: one token, every set's drafts (orig first; copies filled from orig)."""
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score

    token, sets, valid = task
    t0 = time.time()
    rows: List[Dict] = []
    try:
        mc = _W["loader"].get_from_token(token)
    except Exception as e:  # noqa: BLE001
        return [dict(set=name, token=token, k=k, error=f"metric cache: {type(e).__name__}: {e}"[:300])
                for name in sets for k in range(len(valid))], dict(load=time.time() - t0, score=0.0, n=0)
    t_load = time.time() - t0
    t1 = time.time()
    n = 0
    done: Dict = {}
    for name, (traj, copy) in sets.items():
        for k in range(len(valid)):
            base = dict(set=name, token=token, k=k)
            if not valid[k]:
                continue
            if copy[k] and ("orig", k) in done:
                rows.append(dict(done[("orig", k)], **base, copied_from_orig=True))
                continue
            try:
                r, _ = pdm_score(mc, Trajectory(np.asarray(traj[k], np.float32), _W["sampling"]),
                                 _W["sim"].proposal_sampling, _W["sim"], _W["scorer"], _W["policy"])
                d = {c: float(r[c].iloc[0]) for c in METRIC_COLS}
                d["score"] = final_score_noec(r)
                d["error"] = None
                n += 1
            except Exception as e:  # noqa: BLE001 -- recorded
                d = {c: np.nan for c in METRIC_COLS}
                d.update(score=np.nan, error=f"{type(e).__name__}: {e}"[:300])
            if name == "orig":
                done[("orig", k)] = d
            rows.append(dict(d, **base, copied_from_orig=False))
    return rows, dict(load=t_load, score=time.time() - t1, n=n)


def run_tasks(tasks, workers: int, log_every: int = 50):
    """-> (rows, timing list, wall seconds); workers <= MAX_WORKERS, fork pool (workers 1 = in process)."""
    from multiprocessing import get_context

    if not 1 <= workers <= MAX_WORKERS:
        raise SystemExit(f"--workers must be in 1..{MAX_WORKERS}")
    cfg = compose_cfg()
    t0 = time.time()
    rows, tim = [], []
    if workers == 1:
        _worker_init(cfg)
        it = map(score_token, tasks)
        pool = None
    else:
        pool = get_context("fork").Pool(workers, initializer=_worker_init, initargs=(cfg,))
        it = pool.imap_unordered(score_token, tasks, chunksize=1)
    for i, (r, t) in enumerate(it):
        rows += r
        tim.append(t)
        if (i + 1) % log_every == 0:
            print(f"[epdms] {i + 1}/{len(tasks)} tokens {time.time() - t0:.0f}s", flush=True)
    if pool is not None:
        pool.close()
        pool.join()
    return rows, tim, time.time() - t0


def make_tasks(S: Dict, set_names: Sequence[str]):
    return [(t, {n: (S["sets"][n]["traj"][i], S["sets"][n]["copy"][i]) for n in set_names}, S["valid"][i])
            for i, t in enumerate(S["tokens"])]


def rows_frame(rows: List[Dict], S: Dict) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    pos = {t: i for i, t in enumerate(S["tokens"])}
    i = df.token.map(pos).to_numpy()
    k = df.k.to_numpy().astype(int)
    df["family"] = S["family"][i, k]
    df["valid"] = S["valid"][i, k]
    df["arm"] = df["set"].map({n: s["arm"] for n, s in S["sets"].items()})
    df["p_g"] = [S["sets"][n]["p_g"][a, b] for n, a, b in zip(df["set"], i, k)]
    return df


# ----------------------------------------------------------------------------------------------- commands
def cmd_bench(a) -> Dict:
    toks_all = np.load(Path(a.runs[0]) / a.eval / "pred.npz")["tokens"]
    toks = pick_tokens(toks_all, a.n_tokens, a.subset_seed)
    S = load_sets([Path(r) for r in a.runs[:1]], a.eval, toks, orig=True, refined=False)
    res = {}
    for w in a.bench_workers:
        rows, tim, wall = run_tasks(make_tasks(S, ["orig"]), w, log_every=10 ** 9)
        n_traj = sum(t["n"] for t in tim)
        busy = sum(t["load"] + t["score"] for t in tim)
        per_traj = sum(t["score"] for t in tim) / max(n_traj, 1)
        load = float(np.mean([t["load"] for t in tim]))
        res[w] = dict(workers=w, n_tokens=len(toks), n_traj=n_traj, wall_s=round(wall, 2),
                      traj_per_s=round(n_traj / wall, 3), per_traj_s=round(per_traj, 4), load_per_token_s=round(load, 4),
                      parallel_eff=round(busy / (w * wall), 3),
                      errors=int(sum(r.get("error") is not None for r in rows)),
                      mean_score_orig=float(np.nanmean([r["score"] for r in rows])))
        print(json.dumps(res[w]), flush=True)
    # extrapolation: all tokens of the eval, 13 drafts, orig + n_arms refined sets per token (one cache load per token)
    n_all = len(set(map(str, toks_all)))
    wbest = max(res)
    r = res[wbest]
    eff = r["parallel_eff"]
    per_token = r["load_per_token_s"] + 13 * (1 + a.n_arms) * r["per_traj_s"]
    est = {}
    for W in (4, 8):
        e = eff if W <= wbest else eff * 0.9           # stated assumption: 10 % extra loss from 4 -> 8 workers
        est[W] = dict(workers=W, full_hours=round(n_all * per_token / (W * e) / 3600, 2), assumed_eff=round(e, 3))
    W8 = est[8]
    budget_s = a.budget_hours * 3600
    n_fit = int(budget_s * 8 * W8["assumed_eff"] / per_token)
    rec = n_all if W8["full_hours"] <= a.budget_hours else int(np.floor(min(n_fit, n_all) / 500) * 500)
    out = dict(bench=res, n_tokens_all=n_all, n_sets=1 + a.n_arms, per_token_s_all_sets=round(per_token, 3),
               estimate=est, budget_hours=a.budget_hours, n_tokens_fit_8w=n_fit, recommended_n_tokens=rec,
               recommendation=("full navtest" if rec == n_all else f"random subset n = {rec} tokens (seed 0), 13 drafts"),
               tokens_sha256=token_hash(toks), created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    print(json.dumps({k: v for k, v in out.items() if k != "bench"}, indent=1), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    return out


def cmd_score(a) -> Path:
    out = Path(a.out)
    runs = [Path(r) for r in a.runs]
    toks_all = np.load(runs[0] / a.eval / "pred.npz")["tokens"]
    toks = pick_tokens(toks_all, a.n_tokens, a.subset_seed)
    check_unique_arms({r.name: run_arm(r) for r in runs})
    S = load_sets(runs, a.eval, toks)
    names = list(S["sets"])
    meta = dict(runs=[str(r) for r in runs], arms={r.name: run_arm(r) for r in runs}, eval=a.eval,
                n_tokens=len(toks), n_tokens_all=len(set(map(str, toks_all))), subset_seed=a.subset_seed,
                n_tokens_arg=a.n_tokens, tokens_sha256=token_hash(toks), sets=names, scorer=str(NAV),
                metric_cache=str(MC_NAVTEST), ec="not computed (two_frame_extended_comfort = NaN, official weight-0 path)")
    out.mkdir(parents=True, exist_ok=True)
    mf = out / "meta.json"
    if mf.exists():
        old = json.loads(mf.read_text())
        for k in ("runs", "eval", "tokens_sha256", "sets"):
            if old.get(k) != meta[k]:
                raise SystemExit(f"{out}: existing meta.json has another {k}; use another --out")
    (out / "tokens.txt").write_text("\n".join(toks) + "\n")
    mf.write_text(json.dumps(meta, indent=1))
    sh = out / "shards"
    sh.mkdir(exist_ok=True)
    chunk = a.chunk
    t0 = time.time()
    for c0 in range(0, len(toks), chunk):
        p = sh / f"chunk_{c0 // chunk:05d}.parquet"
        if p.exists():
            continue
        sub = dict(S, tokens=S["tokens"][c0:c0 + chunk], valid=S["valid"][c0:c0 + chunk],
                   family=S["family"][c0:c0 + chunk],
                   sets={n: {kk: (vv[c0:c0 + chunk] if isinstance(vv, np.ndarray) else vv) for kk, vv in s.items()}
                         for n, s in S["sets"].items()})
        rows, tim, wall = run_tasks(make_tasks(sub, names), a.workers)
        df = rows_frame(rows, sub)
        tmp = p.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(p)
        print(f"[epdms] chunk {c0 // chunk} ({len(sub['tokens'])} tokens) {wall:.0f}s, total {time.time() - t0:.0f}s",
              flush=True)
    df = pd.concat([pd.read_parquet(p) for p in sorted(sh.glob("chunk_*.parquet"))], ignore_index=True)
    df.to_parquet(out / "rows.parquet", index=False)
    meta.update(n_rows=len(df), n_errors=int(df.error.notna().sum()), finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    mf.write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "runs"}), flush=True)
    return out


def final_frame(df: pd.DataFrame, set_name: str, theta: float) -> pd.DataFrame:
    """Per valid draft: final = the set's tau1 row if p_g >= theta else the orig row.

    df = valid rows INCLUDING errored ones (error filtering happens here).  A draft is dropped when its orig row is
    missing / errored / non-finite, or when it is modified (p_g >= theta) and its refined row is missing / errored /
    non-finite -- it is never silently scored as the original.  p_g comes from the set's own row even when that row
    errored (rows_frame stores p_g on every row).  Drop counts are returned in fin.attrs:
    n_orig_errors, n_refined_errors (modified drafts whose refined score is unusable)."""
    cols = list(METRIC_COLS) + ["score"]
    o_all = df[df["set"] == "orig"].set_index(["token", "k"])
    r = df[df["set"] == set_name].set_index(["token", "k"]).reindex(o_all.index)
    o_ok = (o_all.error.isna() & np.isfinite(o_all[cols].to_numpy(np.float64)).all(1)).to_numpy()
    mod = (r.p_g >= theta).to_numpy()
    r_ok = (r.error.isna() & np.isfinite(r[cols].to_numpy(np.float64)).all(1)).to_numpy()   # missing -> NaN -> False
    ref_err = mod & ~r_ok
    fin = pd.DataFrame(np.where(mod[:, None], r[cols].to_numpy(np.float64), o_all[cols].to_numpy(np.float64)),
                       index=o_all.index, columns=cols)
    fin["score_orig"] = o_all["score"].to_numpy(np.float64)
    fin["modified"] = mod
    fin["family"] = o_all["family"].to_numpy()
    keep = o_ok & ~ref_err
    out = fin[keep].reset_index()
    out.attrs.update(n_orig_errors=int((~o_ok).sum()), n_refined_errors=int((o_ok & ref_err).sum()))
    return out


def check_unique_arms(arms: Dict[str, str]) -> None:
    """report keys results by arm (one run per arm); several runs of one arm (seeds) are refused, not overwritten."""
    seen: Dict[str, List[str]] = {}
    for run, arm in arms.items():
        seen.setdefault(arm, []).append(run)
    dup = {a: r for a, r in seen.items() if len(r) > 1}
    if dup:
        raise SystemExit(f"several runs of the same arm ({dup}); EPDMS report takes one run per arm (no seed averaging)")


def cmd_report(a) -> Dict:
    d = Path(a.scores)
    df = pd.read_parquet(d / "rows.parquet")
    meta = json.loads((d / "meta.json").read_text())
    check_unique_arms(meta["arms"])
    df = df[df.valid]                                    # errored rows are handled per draft in final_frame
    sel = json.loads(Path(a.selection).read_text()) if a.selection else None
    per, fins = {}, {}
    for run, arm in meta["arms"].items():
        th = float(a.theta) if a.theta is not None else sel["arms"][arm]["theta"]
        f = final_frame(df, run, th)
        fins[arm] = f
        per[arm] = dict(run=run, theta=th, n=len(f), n_orig_errors=f.attrs["n_orig_errors"],
                        n_refined_errors=f.attrs["n_refined_errors"], epdms=100 * float(f.score.mean()),
                        epdms_orig=100 * float(f.score_orig.mean()), d_epdms_points=100 * float((f.score - f.score_orig).mean()),
                        modified_frac=float(f.modified.mean()),
                        **{f"{c}_mean": float(f[c].mean()) for c in METRIC_COLS})
    pairs = {}
    logs = None
    if a.pairs:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        import stageT_decision as SD

        run0 = Path(meta["runs"][0])
        tk = pd.read_parquet(run0 / meta["eval"] / "tokens.parquet")[["token", "log"]].drop_duplicates("token")
        logs = dict(zip(tk.token, tk.log))
        for p in a.pairs.split(","):
            x, y = p.split(":")
            m = fins[x].merge(fins[y], on=["token", "k"], suffixes=("_A", "_B"))
            r = SD.cluster_boot((m.score_A - m.score_B).to_numpy(np.float64), m.token.map(logs).to_numpy(), a.n_boot,
                                level=a.ci_level)
            pairs[p] = {k: (100 * v if k in ("mean", "lo", "hi") else v) for k, v in r.items()}
    res = dict(scores=str(d), n_tokens=meta["n_tokens"], n_tokens_all=meta["n_tokens_all"],
               subset_seed=meta["subset_seed"], ec=meta["ec"], arms=per, pairs_epdms_points=pairs,
               ci_level=a.ci_level, note="reported, not decisive (AMENDMENT 6 addendum)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    return res


def get_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["bench", "score", "report"])
    ap.add_argument("--runs", nargs="*", default=[], help="run directories (their <eval>/pred.npz)")
    ap.add_argument("--eval", default="eval_navtest")
    ap.add_argument("--n-tokens", type=int, default=None, help="random token subset size (default: all tokens)")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bench-workers", type=int, nargs="*", default=[1, 4])
    ap.add_argument("--n-arms", type=int, default=4, help="bench: refined sets per token in the full-run estimate")
    ap.add_argument("--budget-hours", type=float, default=2.0)
    ap.add_argument("--chunk", type=int, default=400, help="score: tokens per resumable shard")
    ap.add_argument("--scores", default=None, help="report: the score --out directory")
    ap.add_argument("--selection", default=None, help="report: stageT_decision select json (theta per arm)")
    ap.add_argument("--theta", type=float, default=None, help="report: one theta for every arm instead of --selection")
    ap.add_argument("--pairs", default=None, help="report: A:B pairs, EPDMS difference with a log-cluster bootstrap CI")
    ap.add_argument("--ci-level", type=float, default=0.9875)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    a = get_parser().parse_args(argv)
    if a.cmd in ("bench", "score"):
        if not a.runs:
            raise SystemExit("--runs is required")
        a.runs = [str(Path(r).resolve()) for r in a.runs]
        if a.out:
            a.out = str(Path(a.out).resolve())
        # absolute paths: the re-exec runs from /tmp
        argv2 = [a.cmd, "--runs", *a.runs, "--eval", a.eval, "--subset-seed", str(a.subset_seed), "--workers",
                 str(a.workers), "--bench-workers", *map(str, a.bench_workers), "--n-arms", str(a.n_arms),
                 "--budget-hours", str(a.budget_hours), "--chunk", str(a.chunk)]
        if a.n_tokens is not None:
            argv2 += ["--n-tokens", str(a.n_tokens)]
        if a.out:
            argv2 += ["--out", a.out]
        ensure_v2(argv2)
        return cmd_bench(a) if a.cmd == "bench" else cmd_score(a)
    if not a.scores:
        raise SystemExit("--scores is required")
    if a.selection is None and a.theta is None:
        raise SystemExit("report needs --selection or --theta")
    return cmd_report(a)


if __name__ == "__main__":
    main()
