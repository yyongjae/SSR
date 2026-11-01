#!/usr/bin/env python
"""Stage-T PDM metric cache for the split tokens (IMPL_SPEC §4), identical in definition to E's and navtest's caches.

What is cached and how (same definition as report/cause_and_correction_tests/E_train_split_feasibility/e2_cache.py):
  navsim.planning.metric_caching.caching.cache_scenarios() -- the function run_metric_caching.py distributes to its
  workers -- called with the *saved* hydra config of the navtest cache (SAVED_CFG: scene filter window 4+10 frames,
  frame_interval 1, has_route, MetricCacheProcessor defaults, force_feature_computation false), overriding only
  cache.cache_path (--out), navsim_log_path (trainval or test logs) and output_dir; scene_filter.log_names / tokens
  are set per chunk inside cache_scenarios (as the repo does per log).
  Provenance: _init() and run_chunk() below are copied from e2_cache.py (sha256 E2_CACHE_SHA256, 2026-09-28) with
  only the return value extended; nothing is imported from E at runtime.  SAVED_CFG's sha256 is asserted at start.
Layout (repo MetricCacheProcessor layout): <out>/<log>/unknown/<token>/metric_cache.pkl  (lzma pickle of MetricCache).
  Default <out> = /home/external-user/ssd/yongjae_refiner/metric_cache.  Loaders: mc_path(), load_mc().

Reuse of E's 9,000 navtrain caches ("seed"): split tokens that have E's cache are COPIED (not symlinked: E's folder
  lives in the repo report tree; a copy on the ssd is robust to clean-ups and faster to read) with an atomic
  tmp-file + os.replace.  E built them with the same e2_cache.py / SAVED_CFG, so they are the same cache by
  definition; the pickled MetricCache.file_path still names E's path (unused by scoring).  `verify-e` re-caches a
  few seeded tokens with this script and checks identity (scores and all cache fields) against the copies.

Resumability: the repo processor skips tokens whose metric_cache.pkl exists.  Because a killed worker can leave a
  truncated pkl, every token whose chunk completed is appended to <out>/_state/done_tokens.txt by the main process;
  at start, pkl files NOT listed there are xz-stream checked (full lzma decompression) and deleted if corrupt, then
  re-cached.  Failed / missing tokens -> <out>/_state/failed_tokens.tsv.

Subcommands (all CPU; run with CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10):
  build          [--splits train,dev] [--workers 4] [--chunk 16]  seed from E, then cache all missing split tokens.
                 Chunks = (log, <= chunk tokens); logs processed in a seeded random order (a partial build covers a
                 random subset of logs of both splits).  Progress log: one line per chunk with rate and ETA.
  verify-navtest [--n 20] [--workers 2]  re-cache n navtest tokens into VERIFY_ROOT/navtest and compare with
                 data/exp/metric_cache: official scores (cf_common.score) of 6 trajectories per token must be identical
                 (network, human, constant-velocity straight, CV with lateral drift, braking CV, PDM-Closed), plus
                 field-by-field equality (ego state, PDM-Closed states, 51 occupancy maps, unique objects, centerline,
                 route, drivable-area map).  -> report/refiner_T/metric_cache_verify_navtest.json
  verify-e       [--n 5]  same comparison for E-cached split tokens re-cached into VERIFY_ROOT/e against E's original
                 files (4 trajectories: CV, CV-lateral, brake, PDM-Closed) + byte equality of the seeded copy.
                 -> report/refiner_T/metric_cache_verify_e.json
  check          xz integrity of every split token's cache -> <out>/manifest.parquet (token, log, split, exists, ok,
                 path) and <out>/metadata/metric_cache_metadata_node_0.csv (navsim MetricCacheLoader format, ok files
                 only).  Run automatically at the end of `build` when no token failed.
  status         counts and ETA from the state files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import pickle
import shutil
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/home/external-user/yongjae/SSR")
sys.path.insert(0, str(ROOT))
SAVED_CFG = ROOT / "data/exp/metric_cache/metadata/code/hydra/config.yaml"
SAVED_CFG_SHA256 = "92d42853253622b29eabf8e488a97741e350eb3d5a360cb45f69cb0a4b444f84"
E2_CACHE_SHA256 = "9e19488582622282a58c4b801f624578640d8b966926de00a769d5ad181cdd9e"
TRAIN_LOGS = ROOT / "data/dataset/navsim_logs/trainval"
TEST_LOGS = Path("/home/external-user/navsim/download/test_navsim_logs/test")
NAVTEST_MC = ROOT / "data/exp/metric_cache"
E_DIR = ROOT / "report/cause_and_correction_tests/E_train_split_feasibility"
E_MC = E_DIR / "metric_cache"
NAVTEST_TABLE = ROOT / "report/perception_reliability/pdm_attr/table.parquet"
CF_DIR = ROOT / "report/collision_counterfactual/counterfactual"
DATA = Path("/home/external-user/ssd/yongjae_refiner")
OUT = DATA / "metric_cache"
VERIFY_ROOT = DATA / "metric_cache_verify"  # re-cached verification tokens (kept OUTSIDE OUT: navtest tokens must never
                                            # be picked up by a glob over the stage-T cache)
SPLITS = DATA / "splits"
REPORT = ROOT / "report/refiner_T"
ENV = dict(NUPLAN_MAPS_ROOT=str(ROOT / "data/dataset/maps"), NUPLAN_MAP_VERSION="nuplan-maps-v1.0",
           OPENSCENE_DATA_ROOT=str(ROOT / "data/dataset"))
SCORE_KEYS = ["nc", "dac", "ddc", "ep", "ttc", "comfort", "pdms"]


# ----------------------------------------------------------------------------------------------- paths / io
def mc_path(root: Path, log: str, token: str) -> Path:
    return Path(root) / log / "unknown" / token / "metric_cache.pkl"


def load_mc(path: Path):
    with lzma.open(path, "rb") as f:
        return pickle.load(f)


def xz_ok(path: Path) -> bool:
    """True iff the file is a complete xz stream (full decompression, no unpickling)."""
    try:
        with lzma.open(path, "rb") as f:
            while f.read(1 << 22):
                pass
        return True
    except Exception:
        return False


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_splits(splits: Sequence[str], root: Path = SPLITS) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(root / f"{s}.parquet", columns=["token", "log"]).assign(split=s) for s in splits])
    assert not df.token.duplicated().any()
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- worker (from e2_cache)
W = {}


def _init(out, logs):
    """load the saved caching config once per worker; only paths are overridden.  scene_filter.log_names /
    tokens are replaced by cache_scenarios() for every chunk anyway (emptied here only to avoid instantiating the
    12k-token navtest list each call).  [copied from e2_cache.py]"""
    for k, v in ENV.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(SAVED_CFG)
    cfg.cache.cache_path = str(out)
    cfg.navsim_log_path = str(logs)
    cfg.output_dir = str(Path(out) / "metadata")
    cfg.scene_filter.tokens = []
    cfg.scene_filter.log_names = []
    W["cfg"] = cfg


def run_chunk(a):
    """[copied from e2_cache.py; returns also the tokens]  a = (log, tokens, out, logs)."""
    log, toks, out, logs = a
    from navsim.planning.metric_caching.caching import cache_scenarios
    t0 = time.time()
    done = [t for t in toks if mc_path(Path(out), log, t).exists()]
    if len(done) == len(toks):
        return log, list(toks), len(toks), 0, 0.0, "cached"
    cfg = W["cfg"]
    try:
        res = cache_scenarios([{"cfg": cfg, "log_file": log, "tokens": list(toks)}])
        ok = sum(r.successes for r in res)
        bad = sum(r.failures for r in res)
        return log, list(toks), ok, bad, time.time() - t0, ""
    except Exception:  # pragma: no cover
        return log, list(toks), 0, len(toks), time.time() - t0, traceback.format_exc()[-600:]


def make_chunks(df: pd.DataFrame, out: Path, logs: Path, chunk: int, seed: int = 0) -> List[tuple]:
    """(log, tokens<=chunk, out, logs) tuples; logs in a seeded random order, tokens sorted inside a log."""
    rng = np.random.default_rng(seed)
    order = sorted(df.log.unique())
    order = [order[i] for i in rng.permutation(len(order))]
    by_log = {lg: sorted(g.token) for lg, g in df.groupby("log")}
    chunks = []
    for lg in order:
        tk = by_log[lg]
        for s in range(0, len(tk), chunk):
            chunks.append((lg, tk[s:s + chunk], str(out), str(logs)))
    return chunks


# ----------------------------------------------------------------------------------------------- state
class State:
    """<out>/_state/{done_tokens.txt, failed_tokens.tsv}: appended by the main process only."""

    def __init__(self, out: Path):
        self.dir = Path(out) / "_state"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.done_f = self.dir / "done_tokens.txt"
        self.fail_f = self.dir / "failed_tokens.tsv"

    def done(self) -> set:
        return set(self.done_f.read_text().split()) if self.done_f.exists() else set()

    def add_done(self, toks: Iterable[str]):
        toks = list(toks)
        if toks:
            with open(self.done_f, "a") as f:
                f.write("\n".join(toks) + "\n")

    def add_failed(self, rows: Iterable[Tuple[str, str, str]]):
        with open(self.fail_f, "a") as f:
            for r in rows:
                f.write("\t".join(r).replace("\n", " ") + "\n")


def seed_from_e(df: pd.DataFrame, out: Path, e_root: Path = E_MC, state: Optional[State] = None) -> int:
    """Copy E's caches of the df tokens that exist in E and not yet in out (atomic tmp + os.replace).  Returns #copied."""
    n = 0
    for tok, lg in zip(df.token, df.log):
        src, dst = mc_path(e_root, lg, tok), mc_path(out, lg, tok)
        if dst.exists() or not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + f".tmp{os.getpid()}")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        n += 1
        if state is not None:
            state.add_done([tok])
    return n


def repair(df: pd.DataFrame, out: Path, state: State) -> Tuple[int, int]:
    """xz-check pkl files of df tokens that exist but are not recorded done; delete corrupt ones, record good ones.
    Returns (#checked, #deleted)."""
    done = state.done()
    checked = deleted = 0
    good = []
    for tok, lg in zip(df.token, df.log):
        p = mc_path(out, lg, tok)
        if tok in done or not p.exists():
            continue
        checked += 1
        if xz_ok(p):
            good.append(tok)
        else:
            p.unlink()
            deleted += 1
    for p in Path(out).glob("*/unknown/*/metric_cache.pkl.tmp*"):
        p.unlink()
    state.add_done(good)
    return checked, deleted


# ----------------------------------------------------------------------------------------------- build
def cmd_build(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assert sha256(SAVED_CFG) == SAVED_CFG_SHA256, "saved caching config changed"
    df = read_splits(a.splits.split(","), Path(a.splits_dir))
    if a.limit:
        df = df.iloc[: a.limit]
    st = State(out)
    t0 = time.time()
    n_seed = 0 if a.no_seed else seed_from_e(df, out, Path(a.e_mc), st)
    checked, deleted = repair(df, out, st)
    pend = df[[not mc_path(out, lg, t).exists() for t, lg in zip(df.token, df.log)]]
    chunks = make_chunks(pend, out, TRAIN_LOGS, a.chunk, a.seed)
    print(f"[{time.strftime('%F %T')}] split tokens {len(df)} | seeded from E {n_seed} | repair checked {checked} "
          f"deleted {deleted} | pending {len(pend)} tokens in {len(chunks)} chunks | workers {a.workers} -> {out}",
          flush=True)
    n_ok = n_bad = 0
    t1 = time.time()
    if chunks:
        with Pool(a.workers, initializer=_init, initargs=(str(out), str(TRAIN_LOGS)), maxtasksperchild=50) as p:
            for log, toks, ok, bad, dt, msg in p.imap_unordered(run_chunk, chunks):
                have = [t for t in toks if mc_path(out, log, t).exists()]
                miss = [t for t in toks if t not in set(have)]
                st.add_done(have)
                if miss:
                    st.add_failed([(t, log, msg[-300:] or "not cached") for t in miss])
                n_ok += len(have)
                n_bad += len(miss)
                el = time.time() - t1
                rate = n_ok / max(el, 1e-6)
                left = len(pend) - n_ok - n_bad
                print(f"[{time.strftime('%T')}] {log} ok {len(have)}/{len(toks)} {dt:.0f}s {msg[:200]} | total ok {n_ok} "
                      f"bad {n_bad} left {left} | {el:.0f}s {rate:.3f} tok/s ETA {left / max(rate, 1e-6) / 3600:.2f} h",
                      flush=True)
    summ = dict(split_tokens=len(df), seeded_from_e=n_seed, repair_checked=checked, repair_deleted=deleted,
                pending_at_start=len(pend), ok=n_ok, bad=n_bad, seconds=time.time() - t0, workers=a.workers,
                finished=time.strftime("%F %T"))
    json.dump(summ, open(st.dir / "build_summary.json", "w"), indent=1)
    print("done", summ, flush=True)
    if n_bad == 0 and not a.limit and not a.no_final_check:
        cmd_check(a)


# ----------------------------------------------------------------------------------------------- comparison
def _states(traj) -> np.ndarray:
    s = traj.get_sampled_trajectory()
    return np.array([[x.time_us, x.rear_axle.x, x.rear_axle.y, x.rear_axle.heading,
                      x.dynamic_car_state.rear_axle_velocity_2d.x, x.dynamic_car_state.rear_axle_velocity_2d.y,
                      x.dynamic_car_state.rear_axle_acceleration_2d.x] for x in s], np.float64)


def _geoms(tokens, geoms) -> Tuple[list, np.ndarray, np.ndarray]:
    import shapely
    coords, idx = shapely.get_coordinates(list(geoms), return_index=True)
    return list(tokens), coords, idx


def _diff_geomset(a, b) -> float:
    """max abs coordinate difference of two (tokens, geometries) sets; inf if tokens / structure differ."""
    ta, ca, ia = _geoms(*a)
    tb, cb, ib = _geoms(*b)
    if ta != tb or ca.shape != cb.shape or not np.array_equal(ia, ib):
        return float("inf")
    return float(np.abs(ca - cb).max()) if len(ca) else 0.0


def compare_fields(mo, mn) -> Dict[str, float]:
    """Field-by-field max abs differences of two MetricCache objects (0.0 = identical; inf = structure differs)."""
    r = {}
    eo, en = mo.ego_state, mn.ego_state
    vo = [eo.rear_axle.x, eo.rear_axle.y, eo.rear_axle.heading, eo.dynamic_car_state.rear_axle_velocity_2d.x,
          eo.dynamic_car_state.rear_axle_velocity_2d.y, eo.dynamic_car_state.rear_axle_acceleration_2d.x,
          eo.dynamic_car_state.rear_axle_acceleration_2d.y, eo.time_us]
    vn = [en.rear_axle.x, en.rear_axle.y, en.rear_axle.heading, en.dynamic_car_state.rear_axle_velocity_2d.x,
          en.dynamic_car_state.rear_axle_velocity_2d.y, en.dynamic_car_state.rear_axle_acceleration_2d.x,
          en.dynamic_car_state.rear_axle_acceleration_2d.y, en.time_us]
    r["ego_state"] = float(np.abs(np.array(vo) - np.array(vn)).max())
    so, sn = _states(mo.trajectory), _states(mn.trajectory)
    r["pdm_closed"] = float(np.abs(so - sn).max()) if so.shape == sn.shape else float("inf")
    oo, on = mo.observation._occupancy_maps, mn.observation._occupancy_maps
    r["obs_n_maps"] = float(abs(len(oo) - len(on)))
    r["obs_maps"] = max((_diff_geomset((a._tokens, a._geometries), (b._tokens, b._geometries)) for a, b in zip(oo, on)),
                        default=0.0) if len(oo) == len(on) else float("inf")
    uo, un = mo.observation._unique_objects, mn.observation._unique_objects
    if sorted(uo) != sorted(un):
        r["unique_objects"] = float("inf")
    else:
        d = 0.0
        for k in uo:
            bo, bn = uo[k].box, un[k].box
            d = max(d, abs(bo.center.x - bn.center.x), abs(bo.center.y - bn.center.y),
                    abs(bo.center.heading - bn.center.heading), abs(bo.length - bn.length), abs(bo.width - bn.width))
        r["unique_objects"] = d
    co, cn = mo.centerline._states_se2_array, mn.centerline._states_se2_array
    r["centerline"] = float(np.abs(co - cn).max()) if co.shape == cn.shape else float("inf")
    r["route"] = 0.0 if list(mo.route_lane_ids) == list(mn.route_lane_ids) else float("inf")
    do, dn = mo.drivable_area_map, mn.drivable_area_map
    r["drivable_map"] = _diff_geomset((do._tokens, do._geometries), (dn._tokens, dn._geometries))
    r["drivable_types"] = 0.0 if list(do._map_types) == list(dn._map_types) else float("inf")
    return r


def probe_trajectories(mc, net: Optional[np.ndarray] = None, human: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Trajectories [8,3] (N frame, t = 0.5..4 s) that exercise different scorer paths."""
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
        convert_absolute_to_relative_se2_array,
    )
    from nuplan.common.actor_state.state_representation import TimePoint
    t = np.arange(1, 9) * 0.5
    v = float(mc.ego_state.dynamic_car_state.speed)
    tr = {}
    if net is not None:
        tr["net"] = np.asarray(net, np.float32)
    if human is not None:
        tr["human"] = np.asarray(human, np.float32)
    tr["cv"] = np.stack([v * t, 0 * t, 0 * t], -1)
    y = 1.5 * (t / 4.0) ** 2
    tr["cv_lat"] = np.stack([v * t, y, np.arctan2(2 * 1.5 * t / 16.0, max(v, 0.5))], -1)
    xb = np.where(v - 2.0 * t > 0, v * t - t ** 2, v ** 2 / 4.0)
    tr["brake"] = np.stack([xb, 0 * t, 0 * t], -1)
    t0 = mc.trajectory.start_time.time_us
    g = np.array([[s.rear_axle.x, s.rear_axle.y, s.rear_axle.heading]
                  for s in (mc.trajectory.get_state_at_time(TimePoint(int(t0 + k * 5e5))) for k in range(1, 9))])
    tr["pdmc"] = convert_absolute_to_relative_se2_array(mc.ego_state.rear_axle, g)
    return {k: np.asarray(x, np.float32) for k, x in tr.items()}


def compare_pair(mo, mn, net=None, human=None) -> dict:
    """Official scores (cf_common.score) of the test trajectories on both caches + field differences."""
    sys.path.insert(0, str(CF_DIR))
    import cf_common as CF
    row = {}
    diffs = []
    for name, tr in probe_trajectories(mo, net, human).items():
        so = CF.score(mo, tr)[0]
        sn = CF.score(mn, tr)[0]
        d = max(abs(so[k] - sn[k]) for k in SCORE_KEYS)
        row[f"score_diff_{name}"] = d
        row[f"pdms_{name}"] = so["pdms"]
        row[f"nc_{name}"] = so["nc"]
        row[f"dac_{name}"] = so["dac"]
        diffs.append(d)
    row["max_score_diff"] = max(diffs)
    f = compare_fields(mo, mn)
    row.update({f"field_{k}": v for k, v in f.items()})
    row["max_field_diff"] = max(f.values())
    return row


def _verify(a, which: str):
    sys.path.insert(0, str(CF_DIR))
    for k, v in ENV.items():
        os.environ.setdefault(k, v)
    import cf_common as CF
    out = Path(a.verify_root) / which
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed + 11)
    if which == "navtest":
        t = pd.read_parquet(NAVTEST_TABLE, columns=["token", "log", "re_no_at_fault_collisions",
                                                    "re_drivable_area_compliance"])
        try:
            prev = {r["token"] for r in json.load(open(E_DIR / "checks/cache_verify.json"))["rows"]}
        except Exception:
            prev = set()
        t = t[~t.token.isin(prev)]
        n_f = a.n * 3 // 10
        nc = t[t.re_no_at_fault_collisions < 1]
        dac = t[(t.re_drivable_area_compliance < 1) & (t.re_no_at_fault_collisions == 1)]
        rest = t[(t.re_no_at_fault_collisions == 1) & (t.re_drivable_area_compliance == 1)]
        pick = pd.concat([nc.iloc[rng.choice(len(nc), n_f, replace=False)],
                          dac.iloc[rng.choice(len(dac), n_f, replace=False)],
                          rest.iloc[rng.choice(len(rest), a.n - 2 * n_f, replace=False)]])
        logs, old_root = TEST_LOGS, NAVTEST_MC
    else:
        df = read_splits(["train", "dev"], Path(a.splits_dir))
        e = df[[mc_path(E_MC, lg, t).exists() for t, lg in zip(df.token, df.log)]]
        pick = e.iloc[rng.choice(len(e), a.n, replace=False)]
        logs, old_root = TRAIN_LOGS, Path(a.e_mc)  # E's original caches (seeded copies are byte copies of these)
    pick = pick[["token", "log"]].reset_index(drop=True)
    chunks = [(r.log, [r.token], str(out), str(logs)) for r in pick.itertuples()]
    t0 = time.time()
    with Pool(min(a.workers, 2), initializer=_init, initargs=(str(out), str(logs))) as p:
        res = list(p.imap_unordered(run_chunk, chunks))
    t_cache = time.time() - t0
    print(f"re-cached {sum(r[2] for r in res)}/{len(chunks)} in {t_cache:.0f}s", flush=True)
    CF.init_worker()
    human = CF.G["RA"].G["human"] if which == "navtest" else {}
    rows = []
    for r in pick.itertuples():
        mo, mn = load_mc(mc_path(old_root, r.log, r.token)), load_mc(mc_path(out, r.log, r.token))
        net = CF.G["traj"][r.token] if which == "navtest" else None
        h = None
        if r.token in human and bool(np.all(human[r.token][1])):
            h = human[r.token][0]
        row = dict(token=r.token, log=r.log, **compare_pair(mo, mn, net, h))
        if which != "navtest":
            cp = mc_path(Path(a.out), r.log, r.token)
            row["seeded_copy_bytes_equal_e"] = bool(cp.exists() and sha256(cp) == sha256(mc_path(old_root, r.log, r.token)))
        rows.append(row)
        print(r.token, "max_score_diff", row["max_score_diff"], "max_field_diff", row["max_field_diff"], flush=True)
    n_traj = sum(1 for r in rows for k in r if k.startswith("score_diff_"))
    s = dict(which=which, n_tokens=len(rows), n_trajectories_scored=n_traj,
             all_scores_identical=all(r["max_score_diff"] == 0 for r in rows),
             all_fields_identical=all(r["max_field_diff"] == 0 for r in rows),
             max_score_diff=max(r["max_score_diff"] for r in rows), max_field_diff=max(r["max_field_diff"] for r in rows),
             n_pdms_below_1=int(sum(r[k] < 1 for r in rows for k in r if k.startswith("pdms_"))),
             n_nc_fail=int(sum(r[k] < 1 for r in rows for k in r if k.startswith("nc_"))),
             n_dac_fail=int(sum(r[k] < 1 for r in rows for k in r if k.startswith("dac_"))),
             recache_seconds=t_cache, old_root=str(old_root), new_root=str(out),
             saved_cfg_sha256=sha256(SAVED_CFG), rows=rows)
    REPORT.mkdir(parents=True, exist_ok=True)
    fn = REPORT / f"metric_cache_verify_{'navtest' if which == 'navtest' else 'e'}.json"
    json.dump(s, open(fn, "w"), indent=1, default=float)
    print({k: v for k, v in s.items() if k != "rows"}, "->", fn)


def cmd_check(a):
    out = Path(a.out)
    df = read_splits(a.splits.split(","), Path(a.splits_dir))
    ex, ok = [], []
    for tok, lg in zip(df.token, df.log):
        p = mc_path(out, lg, tok)
        e = p.exists()
        ex.append(e)
        ok.append(e and xz_ok(p))
    df["exists"], df["ok"] = ex, ok
    df["path"] = [str(mc_path(out, lg, t)) for t, lg in zip(df.token, df.log)]
    df.to_parquet(out / "manifest.parquet", index=False)
    (out / "metadata").mkdir(exist_ok=True)
    with open(out / "metadata" / "metric_cache_metadata_node_0.csv", "w") as f:
        f.write("file_name\n" + "".join(p + "\n" for p in df.path[df.ok]))
    print(f"[{time.strftime('%F %T')}] check:", df.groupby("split")[["exists", "ok"]].sum().to_dict(), "of",
          df.groupby("split").size().to_dict(), flush=True)


def cmd_status(a):
    out = Path(a.out)
    df = read_splits(a.splits.split(","), Path(a.splits_dir))
    st = State(out)
    done = st.done() & set(df.token)
    nf = len(st.fail_f.read_text().splitlines()) if st.fail_f.exists() else 0
    print(f"done {len(done)}/{len(df)} failed-lines {nf}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["build", "verify-navtest", "verify-e", "check", "status"])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--splits", default="train,dev")
    ap.add_argument("--splits-dir", default=str(SPLITS))
    ap.add_argument("--e-mc", default=str(E_MC))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-seed", action="store_true", help="do not copy E's caches")
    ap.add_argument("--no-final-check", action="store_true", help="skip the integrity check / manifest after build")
    ap.add_argument("--verify-root", default=str(VERIFY_ROOT))
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    if a.workers > 4:
        raise SystemExit("at most 4 workers (shared machine)")
    for k, v in ENV.items():
        os.environ.setdefault(k, v)
    {"build": cmd_build, "verify-navtest": lambda x: _verify(x, "navtest"), "verify-e": lambda x: _verify(x, "e"),
     "check": cmd_check, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
