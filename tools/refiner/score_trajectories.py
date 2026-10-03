#!/usr/bin/env python
"""Official PDM scoring of K trajectories per token in ONE simulate/score call (stage T, IMPL_SPEC §3.3).

score_token(metric_cache, trajs[K, 8, 3]) simulates the proposals [PDM-Closed, traj_1..K] together (one
PDMSimulator.simulate_proposals + one PDMScorer.score_proposals call) and returns, per trajectory, exactly the
values that the single-trajectory official ``navsim.evaluate.pdm_score.pdm_score`` (proposals [PDM-Closed, traj_k],
submitted trajectory = proposal index 1) returns for that trajectory alone.  Equality is BITWISE (float ==), checked
by tools/refiner/tests/test_scorer.py and tools/refiner/check_scorer_equivalence.py.

Conventions
  trajs      : [K, 8, 3] (x, y, heading) at t = 0.5 .. 4.0 s in the N frame (NAVSIM ego frame at t0: rear-axle
               origin, x forward, y left, heading CCW from +x, rad; metres).  Cast to float32 first, exactly like the
               official agent output (navsim Trajectory poses are float32) and cf_common.score.
  proposals  : index 0 = PDM-Closed (metric_cache.trajectory), index i = trajs[i - 1]; 41 states at 0.1 s (0 .. 4 s),
               LQR + kinematic bicycle tracked (PDMSimulator), StateIndex layout (11 values, global map frame).
  time idx   : n = 0 .. 40 (t = 0.1 n s) for nc_time_idx / ttc_time_idx / dac_time_idx; -1 = no event.

Scorer configuration actually in effect (verified, see tests)
  navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml -- the file run_pdm_score.py composes
  (default_run_pdm_score.yaml defaults) -- instantiated with hydra.  progress_distance_threshold = 5.0 m there; the
  PDMScorerConfig dataclass default (0.1 m) is NOT what the official evaluation uses.  The archived eval hydra config
  that cf_common / rescore_attr instantiate (work_dirs/eval/para_ssr_interaction_final/code/hydra/config.yaml) has the
  identical PDMScorerConfig (asserted in test_scorer.py), so cf_common.score == pdm_score with this config.

Why a plain score_proposals on K+1 proposals is NOT the official score (critic_impl.md issue 5)
  * DDC: pdm_scorer._calculate_driving_direction_compliance (L398-439) builds
        [sum(oncoming_progress[max(0, i - horizon): i + 1]) for i in range(len(oncoming_progress))]
    where the rows of oncoming_progress are PROPOSALS, i.e. it sums over the proposal axis, not over time.  In the
    official 2-proposal call this gives DDC_traj = thr(max_t(op_pdm[t] + op_traj[t])) (PDM-Closed's oncoming
    progress is added).  In a K+1 batch row i would sum rows i-10..i.  -> recomputed per trajectory on the 2-row
    pair [op_0, op_i] with the verbatim official expression and thresholds.
  * EP: _aggregate_scores normalises raw*mult by the max over ALL proposals.  Official = pair max:
        m_i = max(raw_0*mult_0, raw_i*mult_i);  EP_i = raw_i*mult_i / m_i  if m_i > progress_distance_threshold
        else (1.0 if mult_i > 0 else 0.0)
    with mult_i recomputed with the fixed DDC_i.
  * PDMS: recomputed with the scorer's own aggregation ops: mult * sum_j(w_j * metric_j) / sum_j w_j.
  NC, DAC, TTC, comfort, raw progress and the ego areas are computed per proposal independently (per-proposal
  collided-track lists, row-wise vector ops) and are taken from the batched call unchanged.

Scorer / recording
  The simulator/scorer are rescore_attr._make_classes(PDMSimulator, PDMScorer) subclasses (imported, verified code,
  used by cf_common.score): the parent (official) NC / TTC methods produce every metric value; the subclass then
  re-runs a verbatim copy of the loops only to record per-event track tokens (rec_ok = the copy reproduced the official
  scores).  nc_track / ttc_track come from those records; nc_time_idx / ttc_time_idx from the official
  _collision_time_idcs / _ttc_time_idcs.

Per-trajectory output (score_token -> list of K dicts)
  nc, dac, ddc, ep, ttc, comfort, pdms      official sub-scores / PDMS (float)
  mult                                      nc * dac * ddc
  raw_progress                              official raw progress of traj [m] (ego-centre projection on the route
                                            centerline, t = 0 -> 4 s, clipped >= 0), NOT multiplied by mult
  pdm_progress_eff                          PDM-Closed raw progress * PDM-Closed mult (token-level; the EP normaliser
                                            candidate; same for all k)
  nc_track, nc_time_idx, nc_obj_type        earliest at-fault collision (ties: agent before static, then first
                                            recorded); '' / -1 if none
  ttc_track, ttc_time_idx                   earliest TTC event (ties: smallest look-ahead step); '' / -1 if none
  dac_time_idx                              first time idx with a corner outside the drivable layers; -1 if none
  ddc_stat_m                                official DDC statistic max_t(op_pdm + op_traj) [m] (see above)
  pdm_nc, pdm_dac, pdm_ddc, pdm_raw_progress PDM-Closed's own values in the same call (token-level)
  rec_ok                                    recording copies reproduced the official NC / TTC arrays
  (states [41, 11] float64 LQR-tracked states of the trajectory, only with return_states=True)
  Extra keys beyond IMPL_SPEC §3.3 (mult, nc_obj_type, ttc_*, dac_time_idx, ddc_stat_m, pdm_*, rec_ok) are additions,
  the §3.3 keys are unchanged.

CLI (resumable, <= 4 workers; one metric-cache load per token)
  python tools/refiner/score_trajectories.py --drafts /home/external-user/ssd/yongjae_refiner/drafts/dev \
      --tokens /home/external-user/ssd/yongjae_refiner/splits/dev.parquet \
      --out /home/external-user/ssd/yongjae_refiner/scores/dev.parquet --workers 4
  --drafts : a directory of <token>.npz, a glob of *.npz (token = file stem), or one packed .npz with arrays
             tokens [N] and <key> [N, K, 8, 3].  Array key --key (default 'drafts'); a per-draft 1-D array 'family'
             [K] in a per-token npz is copied to the output.
  --tokens : parquet with column token (+ optional log, used to locate the metric cache directly); default = all
             tokens that have drafts.
  Output   : one row per (token, k) with columns token, log, k, <score_token keys>, family (if in the npz), error,
             sec_load, sec_score (per token); shards <out stem>.shards/part-*.parquet are written as they finish; a rerun
             skips tokens already present (tokens whose metric cache / drafts are missing are not written and are
             retried; scoring exceptions are written as rows with k = -1 and 'error', retried with --retry-errors).
             When all tokens are done the shards are merged into --out and <out>.meta.json records the config,
             source sha256 and timing.
  Metric caches are searched as <root>/<log>/unknown/<token>/metric_cache.pkl in --mc-roots (default: the stage-T
  cache, E's navtrain cache, the navtest cache, in that order).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import lzma
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("NUPLAN_MAPS_ROOT", "/home/external-user/yongjae/SSR/data/dataset/maps")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

ROOT = Path("/home/external-user/yongjae/SSR")
PDM_ATTR = ROOT / "report/perception_reliability/pdm_attr"
for _p in (str(ROOT), str(PDM_ATTR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA = Path("/home/external-user/ssd/yongjae_refiner")
SCORING_YAML = ROOT / "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml"
ARCHIVED_EVAL_CFG = ROOT / "work_dirs/eval/para_ssr_interaction_final/code/hydra/config.yaml"
MC_ROOTS = (
    DATA / "metric_cache",
    ROOT / "report/cause_and_correction_tests/E_train_split_feasibility/metric_cache",
    ROOT / "data/exp/metric_cache",
)
SOURCES = (
    Path(__file__).resolve(),
    PDM_ATTR / "rescore_attr.py",
    ROOT / "navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py",
    ROOT / "navsim/evaluate/pdm_score.py",
    SCORING_YAML,
)
NT = 41  # tracked states 0 .. 4.0 s at 0.1 s
OUT_KEYS = (
    "nc", "dac", "ddc", "ep", "ttc", "comfort", "pdms", "mult", "raw_progress", "pdm_progress_eff",
    "nc_track", "nc_time_idx", "nc_obj_type", "ttc_track", "ttc_time_idx", "dac_time_idx", "ddc_stat_m",
    "pdm_nc", "pdm_dac", "pdm_ddc", "pdm_raw_progress", "rec_ok",
)

_G: Dict[str, object] = {}


# ----------------------------------------------------------------------------------------------- scorer objects
def build_simulator_scorer(record: bool = True, cfg_path: Path = SCORING_YAML):
    """(simulator, scorer) instantiated from the official scoring yaml (hydra).  record=True -> rescore_attr's
    recording subclasses (official values + per-event records), False -> the plain navsim classes."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(cfg_path)
    sub = OmegaConf.create({"proposal_sampling": cfg.proposal_sampling, "simulator": cfg.simulator,
                            "scorer": cfg.scorer})
    sim = instantiate(sub.simulator)
    scorer = instantiate(sub.scorer)
    assert sim.proposal_sampling == scorer.proposal_sampling
    if record:
        import rescore_attr as RA
        rsim_cls, rscorer_cls = RA._make_classes(type(sim), type(scorer))
        sim = rsim_cls(sim.proposal_sampling)
        scorer = rscorer_cls(scorer.proposal_sampling, scorer._config, scorer._vehicle_parameters)
    return sim, scorer


def get_simulator_scorer():
    """Process-wide cached recording (simulator, scorer)."""
    if "sim" not in _G:
        _G["sim"], _G["scorer"] = build_simulator_scorer(record=True)
    return _G["sim"], _G["scorer"]


# ----------------------------------------------------------------------------------------------- metric cache
def load_metric_cache(path):
    with lzma.open(path, "rb") as f:
        return pickle.load(f)


def locate_metric_cache(token: str, log: Optional[str] = None, roots: Sequence[Path] = MC_ROOTS,
                        index: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Path of <root>/<log>/unknown/<token>/metric_cache.pkl (first root that has it), or via ``index``
    (token -> path, from index_metric_cache) when log is unknown.  None if absent."""
    if log:
        for r in roots:
            p = Path(r) / log / "unknown" / token / "metric_cache.pkl"
            if p.exists():
                return str(p)
    if index is not None:
        return index.get(token)
    return None


def index_metric_cache(roots: Sequence[Path] = MC_ROOTS, tokens: Optional[Iterable[str]] = None) -> Dict[str, str]:
    """token -> metric_cache.pkl path over all roots (earlier roots win).  ``tokens`` limits the result."""
    want = set(tokens) if tokens is not None else None
    idx: Dict[str, str] = {}
    for r in roots:
        r = Path(r)
        if not r.is_dir():
            continue
        for log_e in os.scandir(r):
            u = Path(log_e.path) / "unknown"
            if not log_e.is_dir() or not u.is_dir():
                continue
            for tok_e in os.scandir(u):
                t = tok_e.name
                if t in idx or (want is not None and t not in want):
                    continue
                p = Path(tok_e.path) / "metric_cache.pkl"
                if p.exists():
                    idx[t] = str(p)
    return idx


# ----------------------------------------------------------------------------------------------- scoring
def proposal_states(metric_cache, trajs: np.ndarray, future_sampling) -> np.ndarray:
    """[K+1, 41, 11] reference states (0.1 s interpolation, global frame): row 0 PDM-Closed, rows 1.. the trajs,
    built with the same official functions pdm_score() uses."""
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    init = metric_cache.ego_state
    samp = TrajectorySampling(num_poses=8, interval_length=0.5)
    rows = [get_trajectory_as_array(metric_cache.trajectory, future_sampling, init.time_point)[None]]
    for d in trajs:
        pred = transform_trajectory(Trajectory(d, samp), init)
        rows.append(get_trajectory_as_array(pred, future_sampling, init.time_point)[None])
    return np.concatenate(rows, axis=0)


def _ddc_value(progress: float, cfg) -> float:
    """Verbatim thresholds of pdm_scorer._calculate_driving_direction_compliance."""
    if progress < cfg.driving_direction_compliance_threshold:
        return 1.0
    elif progress < cfg.driving_direction_violation_threshold:
        return 0.5
    return 0.0


def pairwise_official(scorer) -> Dict[str, np.ndarray]:
    """From a scorer that has just run score_proposals on [PDM-Closed, traj_1..K], the per-proposal values that the
    official 2-proposal call [PDM-Closed, traj_i] would give for traj_i (entries 1..K; entry 0 = PDM-Closed's own
    DDC / mult, EP / PDMS of entry 0 are NaN).  Uses the scorer's own float operations in the same order."""
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
        BBCoordsIndex, EgoAreaIndex, MultiMetricIndex, WeightedMetricIndex,
    )

    cfg = scorer._config
    P = scorer._num_proposals
    # --- DDC (official expression on each 2-row pair [op_0, op_i])
    center = scorer._ego_coords[:, :, BBCoordsIndex.CENTER]
    op = np.zeros((P, scorer.proposal_sampling.num_poses + 1), dtype=np.float64)
    op[:, 1:] = ((center[:, 1:] - center[:, :-1]) ** 2.0).sum(axis=-1) ** 0.5
    op[~scorer._ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC]] = 0.0
    horizon = int(cfg.driving_direction_horizon / scorer.proposal_sampling.interval_length)
    ddc = np.empty(P, dtype=np.float64)
    stat = np.empty(P, dtype=np.float64)
    for i in range(P):
        pair = op[[0]] if i == 0 else op[[0, i]]
        over = np.array([sum(pair[max(0, j - horizon): j + 1]) for j in range(len(pair))], dtype=np.float64)
        stat[i] = over.max(axis=-1)[-1]
        ddc[i] = _ddc_value(stat[i], cfg)
    mm = scorer._multi_metrics.copy()
    mm[MultiMetricIndex.DRIVING_DIRECTION] = ddc
    mult = mm.prod(axis=0)
    # --- EP (pair normalisation, official threshold rule)
    raw_eff = scorer._progress_raw * mult
    ep = np.full(P, np.nan, dtype=np.float64)
    for i in range(1, P):
        pair_raw = np.array([raw_eff[0], raw_eff[i]], dtype=np.float64)
        pair_mult = np.array([mult[0], mult[i]], dtype=np.float64)
        m = np.max(pair_raw)
        if m > cfg.progress_distance_threshold:
            norm = pair_raw / m
        else:
            norm = np.ones(2, dtype=np.float64)
            norm[pair_mult == 0.0] = 0.0
        ep[i] = norm[1]
    # --- PDMS (scorer's aggregation ops)
    wm = scorer._weighted_metrics.copy()
    wm[WeightedMetricIndex.PROGRESS] = ep
    warr = cfg.weighted_metrics_array
    weighted = (wm * warr[..., None]).sum(axis=0)
    weighted /= warr.sum()
    pdms = mm.prod(axis=0) * weighted
    return dict(nc=mm[MultiMetricIndex.NO_COLLISION], dac=mm[MultiMetricIndex.DRIVABLE_AREA], ddc=ddc,
                ep=ep, ttc=wm[WeightedMetricIndex.TTC], comfort=wm[WeightedMetricIndex.COMFORTABLE], pdms=pdms,
                mult=mult, raw=scorer._progress_raw.copy(), raw_eff=raw_eff, ddc_stat=stat)


def _idx(v) -> int:
    return int(v) if np.isfinite(v) else -1


def score_token(metric_cache, trajs, sim=None, scorer=None, return_states: bool = False) -> List[dict]:
    """Official PDM sub-scores of each of the K trajectories (see module docstring for keys / conventions).

    metric_cache : navsim MetricCache of the token.
    trajs        : [K, 8, 3] N-frame poses at t = 0.5 .. 4.0 s (cast to float32).
    sim, scorer  : from build_simulator_scorer(record=True); default = process-wide cached pair.
    """
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import EgoAreaIndex

    if sim is None or scorer is None:
        sim, scorer = get_simulator_scorer()
    trajs = np.asarray(trajs, dtype=np.float32)
    if trajs.ndim == 2:
        trajs = trajs[None]
    assert trajs.ndim == 3 and trajs.shape[1:] == (8, 3) and len(trajs) >= 1, trajs.shape
    assert np.isfinite(trajs).all(), "non-finite trajectory"
    states = proposal_states(metric_cache, trajs, sim.proposal_sampling)
    simulated = sim.simulate_proposals(states, metric_cache.ego_state)
    scorer.score_proposals(simulated, metric_cache.observation, metric_cache.centerline,
                           metric_cache.route_lane_ids, metric_cache.drivable_area_map)
    v = pairwise_official(scorer)
    has_rec = hasattr(scorer, "rec_nc_events")  # plain PDMScorer: no records -> tracks '', rec_ok True
    rec_ok = bool(getattr(scorer, "rec_nc_consistent", True) and getattr(scorer, "rec_ttc_consistent", True))
    nc_ev = getattr(scorer, "rec_nc_events", [])
    ttc_ev = getattr(scorer, "rec_ttc_events", [])
    offroad = scorer._ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA]
    out = []
    for i in range(1, len(trajs) + 1):
        nc_t = _idx(scorer._collision_time_idcs[i])
        ttc_t = _idx(scorer._ttc_time_idcs[i])
        af = [e for e in nc_ev if e["proposal"] == i and e["at_fault"] and e["time_idx"] == nc_t]
        af = sorted(af, key=lambda e: e["value"])  # stable: agent (0.0) before static (0.5), then record order
        tt = sorted([e for e in ttc_ev if e["proposal"] == i and e["time_idx"] == ttc_t],
                    key=lambda e: e["future_idx"])
        off = np.flatnonzero(offroad[i])
        d = dict(
            nc=float(v["nc"][i]), dac=float(v["dac"][i]), ddc=float(v["ddc"][i]), ep=float(v["ep"][i]),
            ttc=float(v["ttc"][i]), comfort=float(v["comfort"][i]), pdms=float(v["pdms"][i]),
            mult=float(v["mult"][i]), raw_progress=float(v["raw"][i]), pdm_progress_eff=float(v["raw_eff"][0]),
            nc_track=af[0]["track"] if af else "", nc_time_idx=nc_t, nc_obj_type=af[0]["obj_type"] if af else "",
            ttc_track=tt[0]["track"] if tt else "", ttc_time_idx=ttc_t,
            dac_time_idx=int(off[0]) if len(off) else -1, ddc_stat_m=float(v["ddc_stat"][i]),
            pdm_nc=float(v["nc"][0]), pdm_dac=float(v["dac"][0]), pdm_ddc=float(v["ddc"][0]),
            pdm_raw_progress=float(v["raw"][0]),
            rec_ok=bool(rec_ok and (not has_rec or ((nc_t < 0 or len(af) > 0) and (ttc_t < 0 or len(tt) > 0)))),
        )
        if return_states:
            d["states"] = np.array(simulated[i], dtype=np.float64, copy=True)
        out.append(d)
    return out


# ----------------------------------------------------------------------------------------------- drafts
class DraftSource:
    """token -> [K, 8, 3] float32 drafts (+ optional 'family' [K]) from a dir / glob / packed npz."""

    def __init__(self, spec: str, key: str = "drafts"):
        self.key = key
        self.packed = None
        self.files: Dict[str, str] = {}
        p = Path(spec)
        if p.is_dir():
            for e in os.scandir(p):
                if e.name.endswith(".npz"):
                    self.files[e.name[:-4]] = e.path
        elif p.is_file() and p.suffix == ".npz" and not any(c in spec for c in "*?["):
            z = np.load(p, allow_pickle=False)
            if "tokens" in z.files:
                self.packed = {str(t): i for i, t in enumerate(z["tokens"])}
                self._arr = np.asarray(z[key], np.float32)  # [N, K, 8, 3], loaded once
            else:
                self.files[p.stem] = str(p)
        else:
            for f in glob.glob(spec):
                if f.endswith(".npz"):
                    self.files[Path(f).stem] = f

    def tokens(self) -> List[str]:
        return sorted(self.packed) if self.packed is not None else sorted(self.files)

    def has(self, token: str) -> bool:
        return token in (self.packed if self.packed is not None else self.files)

    def get(self, token: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.packed is not None:
            return self._arr[self.packed[token]], None
        with np.load(self.files[token], allow_pickle=False) as z:
            d = np.asarray(z[self.key], np.float32)
            fam = np.asarray(z["family"]) if "family" in z.files else None
        if fam is not None and fam.shape != (len(d),):
            fam = None
        return d, fam


# ----------------------------------------------------------------------------------------------- CLI workers
_W: Dict[str, object] = {}


def _worker_init(draft_spec: str, key: str, roots: List[str]):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _W["src"] = DraftSource(draft_spec, key)
    _W["roots"] = [Path(r) for r in roots]
    get_simulator_scorer()


def _score_shard(task):
    """task = (shard_path, [(token, log or '', mc_path or '')]) -> (shard_path, n_ok, n_missing, n_err, seconds)."""
    import pandas as pd

    shard, items = task
    t0 = time.time()
    rows, n_ok, n_miss, n_err = [], 0, 0, 0
    src = _W["src"]
    for tok, log, mcp in items:
        if not mcp:
            mcp = locate_metric_cache(tok, log or None, _W["roots"]) or ""
        if not mcp or not src.has(tok):
            n_miss += 1
            continue
        log = log or Path(mcp).parent.parent.parent.name  # <root>/<log>/unknown/<token>/metric_cache.pkl
        try:
            trajs, fam = src.get(tok)
            ts = time.time()
            mc = load_metric_cache(mcp)
            tl = time.time()
            res = score_token(mc, trajs)
            te = time.time()
            for k, r in enumerate(res):
                r = dict(token=tok, log=log, k=k, **r, error="", sec_load=tl - ts, sec_score=te - tl)
                if fam is not None:
                    r["family"] = int(fam[k])
                rows.append(r)
            n_ok += 1
        except Exception as e:  # deterministic failures are recorded, retried with --retry-errors
            import traceback
            rows.append(dict(token=tok, log=log, k=-1,
                             error=(repr(e) + " | " + traceback.format_exc()[-400:])[:800]))
            n_err += 1
    if rows:
        df = pd.DataFrame(rows)
        tmp = str(shard) + ".tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, shard)
    return str(shard), n_ok, n_miss, n_err, time.time() - t0


def _sha256(p: Path) -> str:
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except OSError:
        return ""


def _done_tokens(shard_dir: Path, retry_errors: bool) -> set:
    import pandas as pd

    done = set()
    for f in sorted(shard_dir.glob("part-*.parquet")):
        d = pd.read_parquet(f, columns=["token", "k"])
        if retry_errors:
            d = d[d.k >= 0]
        done.update(d.token.tolist())
    return done


def merge_shards(shard_dir: Path, out: Path):
    import pandas as pd

    parts = [pd.read_parquet(f) for f in sorted(shard_dir.glob("part-*.parquet"))]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(df):
        # a retried error token has both an error row (k=-1) and score rows: keep the latest outcome per token
        ok_tok = set(df.loc[df.k >= 0, "token"])
        df = df[(df.k >= 0) | ~df.token.isin(ok_tok)]
        df = df.drop_duplicates(["token", "k"], keep="last").sort_values(["token", "k"]).reset_index(drop=True)
    tmp = str(out) + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    return df


def main(argv=None):
    import pandas as pd
    from multiprocessing import Pool

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--drafts", required=True, help="dir of <token>.npz | glob | packed npz (tokens, <key>)")
    ap.add_argument("--tokens", default="", help="parquet with token (+ log); default: all draft tokens")
    ap.add_argument("--out", required=True, help="output parquet (shards go to <out stem>.shards/)")
    ap.add_argument("--key", default="drafts")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard-size", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mc-roots", nargs="*", default=[str(r) for r in MC_ROOTS])
    ap.add_argument("--retry-errors", action="store_true")
    a = ap.parse_args(argv)
    assert 1 <= a.workers <= 4, "machine is shared: at most 4 workers"

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = out.parent / (out.stem + ".shards")
    shard_dir.mkdir(exist_ok=True)
    src = DraftSource(a.drafts, a.key)
    if a.tokens:
        tdf = pd.read_parquet(a.tokens)
        toks = tdf.token.astype(str).tolist()
        logs = tdf.log.astype(str).tolist() if "log" in tdf.columns else [""] * len(toks)
    else:
        toks = src.tokens()
        logs = [""] * len(toks)
    if a.limit:
        toks, logs = toks[: a.limit], logs[: a.limit]
    done = _done_tokens(shard_dir, a.retry_errors)
    todo = [(t, l) for t, l in zip(toks, logs) if t not in done]
    roots = [Path(r) for r in a.mc_roots]
    idx = None
    if any(not l for _, l in todo):
        idx = index_metric_cache(roots, [t for t, l in todo if not l])
    items = [(t, l, locate_metric_cache(t, l or None, roots, idx) or "") for t, l in todo]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tasks = [(shard_dir / f"part-{stamp}-{i // a.shard_size:05d}.parquet", items[i:i + a.shard_size])
             for i in range(0, len(items), a.shard_size)]
    print(f"{len(toks)} tokens, {len(done)} already scored, {len(todo)} to do in {len(tasks)} shards, "
          f"workers {a.workers} -> {out}", flush=True)
    t0 = time.time()
    tot = dict(ok=0, miss=0, err=0)
    if tasks:
        with Pool(a.workers, initializer=_worker_init, initargs=(a.drafts, a.key, [str(r) for r in roots]),
                  maxtasksperchild=50) as pool:
            for j, (sh, ok, miss, err, dt) in enumerate(pool.imap_unordered(_score_shard, tasks)):
                tot["ok"] += ok
                tot["miss"] += miss
                tot["err"] += err
                el = time.time() - t0
                rate = tot["ok"] / max(el, 1e-9)
                left = len(todo) - tot["ok"] - tot["miss"] - tot["err"]
                print(f"[{j + 1}/{len(tasks)}] {Path(sh).name} ok {ok} missing {miss} err {err} {dt:.0f}s | total ok "
                      f"{tot['ok']} missing {tot['miss']} err {tot['err']} {el:.0f}s {rate:.2f} tok/s "
                      f"ETA {left / max(rate, 1e-9) / 60:.1f} min", flush=True)
    done = _done_tokens(shard_dir, False)
    complete = all(t in done for t in toks)
    meta = dict(
        drafts=a.drafts, tokens=a.tokens, out=str(out), n_tokens=len(toks), n_scored_tokens=len(done & set(toks)),
        complete=complete, run=tot, seconds=time.time() - t0, workers=a.workers, mc_roots=[str(r) for r in roots],
        scorer_yaml=str(SCORING_YAML), scorer_config=repr(get_simulator_scorer()[1]._config),
        sha256={str(p): _sha256(p) for p in SOURCES},
    )
    if complete:
        df = merge_shards(shard_dir, out)
        meta["rows"] = int(len(df))
        meta["n_error_tokens"] = int((df.k < 0).sum()) if len(df) else 0
        print(f"merged {len(df)} rows -> {out}", flush=True)
    else:
        print(f"incomplete ({len(done & set(toks))}/{len(toks)} tokens scored; missing metric cache or drafts) -> "
              f"shards kept in {shard_dir}; rerun to resume", flush=True)
    json.dump(meta, open(str(out) + ".meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
