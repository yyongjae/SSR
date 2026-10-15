#!/usr/bin/env python
"""Draft bank (IMPL_SPEC §3.6): K = 13 perturbed-human / constant-velocity drafts per token + official labels.

Frames / units: N frame (NAVSIM ego frame at t0: rear-axle origin, x forward, y left, heading CCW from +x, rad);
a draft is [8, 3] float32 = (x, y, heading) at t = 0.5 .. 4.0 s with the implicit origin pose at t = 0; metres,
seconds, m/s, m/s^2.

Inputs
  human/<split>.npz (extract_human.py): traj [8,3] (the human trajectory = identity draft bytes), path [16,3] (human
      poses to 8 s, the path source of speed-ups; rows >= n_reg are treated as unavailable), n_reg, v0, a0,
      frame_gap, tokens, logs.  Tokens with frame_gap (a 1.0 s step among future frames 1..10) are SKIPPED.
  metric cache <root>/<log>/unknown/<token>/metric_cache.pkl (score_trajectories.MC_ROOTS): needed for the labels,
      so drafts are generated only for tokens whose cache exists and was last written >= --min-age s ago (the caching
      job may still be writing it).  The route centerline (metric_cache.centerline, global -> N with navsim's
      convert_absolute_to_relative_se2_array about metric_cache.ego_state.rear_axle, the scorer's origin) is loaded
      only for L-creep candidates (human S_8 < 2 m; see needs_centerline for the exploration option):
      decoder.centerline_continuation continues the creep draft on it when the human path is too short.

Slots (ids fixed = decoder.BANK_LAYOUT; family codes = decoder.FAMILY)
  0 identity | 1-3 small | 4-6 lconst | 7 ignore_brake | 8 creep | 9-10 lat | 11 combined | 12 cv
  Each slot is decoder.sample_perturbation(family, ctx, rng) with decoder.sample_bank's fallback rules:
  ignore_brake -> lconst if the human decelerates < 1 m/s; creep -> lconst if human S_8 >= 2 m; lat / combined ->
  lconst if S_8 < 3 m; a rejected lat / combined / ignore_brake / creep -> lconst; a rejected cv -> hdrift; a draft
  still rejected keeps the human bytes with valid = False.  The loop is re-implemented here (make_bank) only so that
  the random generator can be wrapped (A_p range, below); with lconst_ap = (0.2, 1.3) make_bank is BITWISE
  decoder.sample_bank (tests/test_make_draft_bank.py).  Seed: decoder.rng_for_token(token) (sha256 of the token).
  Every perturbed draft is produced by the M4 decoder (mode 'P') in the refiner's own basis, so t0 continuity is
  structural (dv(0) = da(0) = 0, d(0) = d'(0) = 0); see decoder.py for the families.

A_p range (IMPL_SPEC §3.6: "tune A_p range on a 500-token pilot" to an official bank failure rate of 20-35 %)
  decoder.sample_perturbation draws the L-const / combined A_p from a hard-coded U[0.2, 1.3].  BankConfig.lconst_ap
  = (lo, hi) replaces exactly that draw: RemapRNG passes every generator call through except uniform(0.2, 1.3),
  which it answers with uniform(lo, hi) (one variate either way, so the stream is unchanged).  The reduced-A_p
  rejection floor (0.2 m/s^2 after the path-length reduction beta) and every other family range are unchanged.
  BankConfig.lat_dp does the same for the lat / combined |D_p| draw (U[0.3, 1.5]); centerline_families is passed
  to sample_perturbation.  Both are EXPLORATION options (spec: only A_p is tuned, only L-creep uses the centerline).
  DEVIATION (pilot, 500 train tokens, report/refiner_T/draft_bank_pilot.json): the 20-35 % target is NOT reachable
  by the A_p range.  Bank failure (NC|DAC|DDC < 1, valid drafts) = 12.3 % at U[0.2, 1.3], 12.5 % at [0.2, 1.6],
  12.9 % at [0.2, 1.96] (decoder bound 0.98 A_UP), 14.0 % at [1.0, 1.96] (valid rate 94.7 -> 91.4 %): L-const
  speed-ups along the human's own path fail only 4-9 %.  The bank is therefore generated with the spec / draft
  range U[0.2, 1.3] (TUNED_AP) and the non-A_p levers are left to the user (centerline speed-ups 14.2 %, |D_p|
  U[0.5, 1.8] 13.3 %, both 15.1 %; TTC-inclusive label 17.4-21.9 %).

Checks (every draft, recomputed on the SAVED float32 bytes)
  first_dv   = |p_1| / 0.5 - v0 (first-segment speed minus ego speed, t0 continuity window [-0.8, +0.6] m/s)
  0.5 s keyframe kinematics (decoder.keyframe_kinematics): lon accel min / max (first one (u_0 - v0) / 0.25 s),
  lateral accel u |dh| / 0.5, |kappa| = |dh| / l (l >= 1 m), yaw rate.
  check  = decoder.check_draft (limits relative to the human: each limit is max(limit, human's own value); lon accel
           [-4.05, 2.40], lateral accel 3.8, |kappa| 0.213, continuity window); '' = pass.  A draft failing it is
           marked valid = False (only hdrift / cv can reach this point; decoded families are checked inside
           sample_perturbation).  abs_ok = the same limits WITHOUT the human-relative relaxation (reported only).
  The official comfort sub-score of each draft (LQR-tracked, Savitzky-Golay) comes with the labels.

Output  drafts/<split>/<token>.npz (numpy savez, allow_pickle=False, written atomically)
  drafts [13,8,3] f32, family [13] i8, params [13,6] f32 (decoder.PARAM_COLS = A_p, t_on, D_p, s_on, aux, ds4;
      A_p = realised accel offset after the path reduction, D_p = realised end offset)     <- IMPL_SPEC §3.6
  slot [13] i8 (designated family of the slot), valid [13] bool, reason [13] str, path_src [13] str
      ('draft' | 'human_long' | 'centerline' | 'extrap'), z_lon [13,6] f32, w_lat [13,6] f32 (raw controls:
      decode(tau_h, z, w, v0, 'P', path=<path_src path>, lat_len=S_8) reproduces a decoded draft),
      checks [13,6] f32 (CHECK_COLS), check [13] str, abs_ok [13] bool, flags [13,len(FLAG_COLS)] f32 (decoder flags
      of decoded drafts, NaN otherwise), token, log, split, v0, a0, S8, S_avail, cfg_json, cfg_hash, version.
  drafts/<split>/_config.json: the BankConfig used; a later run with a different config refuses (no mixing).
Labels  scores/<split>.parquet: score_trajectories.py on drafts/<split> (one row per (token, k), family copied from
  the npz; resumable shards in scores/<split>.shards/, merged whenever every drafted token is scored).
  Failure label (IMPL_SPEC §3.7 gate): fail = NC < 1 or DAC < 1 or DDC < 1; fail_ttc additionally TTC < 1.

CLI (CPU only: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10; <= 2 workers)
  make_draft_bank.py pilot --tag default --ap 0.2 1.3 [--n 500 --seed 0 --workers 2]
      500 random non-gap TRAIN tokens with a ready metric cache (drafts/_pilot/tokens.parquet, fixed once drawn)
      -> drafts/_pilot/<tag>/ -> scores/_pilot/<tag>.parquet -> report/refiner_T/draft_bank_pilot_<tag>.json
  make_draft_bank.py compare --tags default tuned            -> report/refiner_T/draft_bank_pilot.json
  make_draft_bank.py generate --split train [--workers 2 --limit N --min-age 120]
  make_draft_bank.py score --split train [--workers 2]
  make_draft_bank.py run --splits train,dev --workers 2 --follow-interval 900 --follow-max-h 12
      passes of (generate -> score) per split until every non-gap token is drafted and scored (or the time limit);
      re-run the same command to resume; finally writes report/refiner_T/draft_bank_<split>.json (summarize).
  make_draft_bank.py summarize --drafts-dir D --scores P --out J
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path("/home/external-user/yongjae/SSR")
TOOLS = ROOT / "tools/refiner"
for _p in (str(ROOT), str(TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from navsim.agents.para_ssr.refiner import decoder as D  # noqa: E402
from navsim.agents.para_ssr.refiner.geometry import KAPPA_MAX, LON_ACC_MAX, LON_ACC_MIN, T_POSE  # noqa: E402

DATA = Path("/home/external-user/ssd/yongjae_refiner")
HUMAN = DATA / "human"
DRAFTS = DATA / "drafts"
SCORES = DATA / "scores"
REPORT = ROOT / "report/refiner_T"
VERSION = "draft_bank_v1"
MAX_WORKERS = 2
DECODER_AP = (0.2, 1.3)          # decoder.sample_perturbation's hard-coded L-const / combined A_p range
DECODER_DP = (0.3, 1.5)          # decoder.sample_perturbation's hard-coded lat / combined |D_p| range
TUNED_AP = (0.2, 1.3)            # default of BankConfig.lconst_ap (set from the pilot; see draft_bank_pilot.json)
CL_NEED_M = 20.0                 # extra centerline_families: load the centerline if S_avail < S_8 + this [m]
CHECK_COLS = ("first_dv", "acc_min", "acc_max", "lat_max", "kappa_max", "yaw_max")
FLAG_COLS = ("alpha", "beta", "lat_on", "ext_m", "extrap_m", "kappa_ratio", "kd_min", "kd_floor_hit",
             "kappa_path_clipped", "cont_ok", "first_seg_dv")
SCORE_COLS = ("nc", "dac", "ddc", "ep", "ttc", "comfort", "pdms")


# ----------------------------------------------------------------------------------------------- config / rng
@dataclass(frozen=True)
class BankConfig:
    """Everything that changes the bank bytes (stored in every npz and in drafts/<split>/_config.json)."""
    lconst_ap: Tuple[float, float] = TUNED_AP
    lat_dp: Tuple[float, float] = DECODER_DP
    mode: str = "A"
    max_tries: int = 8
    centerline_families: Tuple[str, ...] = ("creep",)
    layout: Tuple[str, ...] = tuple(D.BANK_LAYOUT)
    version: str = VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "BankConfig":
        d = json.loads(s)
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:12]


class RemapRNG:
    """numpy Generator proxy: uniform(lo, hi) calls whose (lo, hi) is a key of ``remap`` draw from the mapped range
    (one variate either way); every other attribute / call is the wrapped generator's.  ``hits`` counts remaps."""

    def __init__(self, rng: np.random.Generator, remap: Dict[Tuple[float, float], Tuple[float, float]]):
        self._rng = rng
        self._remap = {(float(a), float(b)): (float(c), float(d)) for (a, b), (c, d) in remap.items()}
        self.hits = 0

    def uniform(self, low=0.0, high=1.0, size=None):
        key = (float(low), float(high)) if size is None and np.ndim(low) == 0 and np.ndim(high) == 0 else None
        if key in self._remap:
            self.hits += 1
            low, high = self._remap[key]
        return self._rng.uniform(low, high, size)

    def __getattr__(self, name):
        return getattr(self._rng, name)


def bank_rng(token: str, cfg: BankConfig):
    rng = D.rng_for_token(token)
    remap = {}
    if tuple(map(float, cfg.lconst_ap)) != DECODER_AP:
        remap[DECODER_AP] = tuple(cfg.lconst_ap)
    if tuple(map(float, cfg.lat_dp)) != DECODER_DP:
        remap[DECODER_DP] = tuple(cfg.lat_dp)
    return RemapRNG(rng, remap) if remap else rng


def needs_centerline(ctx: "D.HumanContext", cfg: BankConfig) -> bool:
    """Load the route centerline for this token?  L-creep candidates (S_8 < 2 m) always; with extra
    centerline_families (an open design option, not the spec default) also when the human path to 8 s is shorter
    than S_8 + CL_NEED_M."""
    if ctx.creep_ok:
        return True
    extra = set(cfg.centerline_families) - {"creep"}
    return bool(extra) and ctx.S_avail < ctx.S8 + CL_NEED_M


# ----------------------------------------------------------------------------------------------- one token
def centerline_n(metric_cache) -> np.ndarray:
    """Route centerline of the metric cache in the N frame: [L, 2] (x, y), points within 200 m of the origin."""
    from nuplan.common.actor_state.state_representation import StateSE2
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
        convert_absolute_to_relative_se2_array,
    )
    ra = metric_cache.ego_state.rear_axle
    st = np.asarray(metric_cache.centerline._states_se2_array, np.float64)
    rel = convert_absolute_to_relative_se2_array(StateSE2(ra.x, ra.y, ra.heading), st.copy())
    keep = np.hypot(rel[:, 0], rel[:, 1]) <= 200.0
    return rel[keep, :2] if keep.sum() >= 2 else rel[:, :2]


def draft_checks(ctx: "D.HumanContext", drafts: np.ndarray):
    """(values [K, 6] f32 (CHECK_COLS), check [K] str (decoder.check_draft, '' = pass), abs_ok [K] bool)."""
    K = len(drafts)
    vals = np.full((K, len(CHECK_COLS)), np.nan, np.float32)
    chk, abs_ok = [], np.zeros(K, bool)
    v0 = ctx.v0
    for k in range(K):
        d = np.asarray(drafts[k], np.float32)
        kin = D.keyframe_kinematics(d, v0)
        fdv = float(np.hypot(float(d[0, 0]), float(d[0, 1])) / T_POSE - v0) if v0 is not None else np.nan
        vals[k] = [fdv, kin["acc_min"], kin["acc_max"], kin["lat_max"], kin["kappa_max"], kin["yaw_max"]]
        chk.append(D.check_draft(ctx, d))
        abs_ok[k] = bool((not np.isfinite(fdv) or D.CONT_LO <= fdv <= D.CONT_HI)
                         and LON_ACC_MIN <= kin["acc_min"] and kin["acc_max"] <= LON_ACC_MAX
                         and kin["lat_max"] <= D.LAT_ACC_PERT and kin["kappa_max"] <= KAPPA_MAX)
    return vals, np.array(chk), abs_ok


def make_bank(tau_h, token: str, *, path_long=None, n_valid=None, v0=None, a0=None,
              centerline: Union[None, np.ndarray, Callable[[], np.ndarray]] = None,
              cfg: Optional[BankConfig] = None) -> Dict:
    """K = len(cfg.layout) drafts for one token (see module docstring).

    centerline: [L, 2] N-frame route centerline, or a callable returning it (called only if needs_centerline(): an
    L-creep candidate, S_8 < 2 m, or a short human path with extra centerline_families), or None.
    Returns dict: drafts [K,8,3] f32, family [K] i8, slot [K] i8, params [K,6] f32, z_lon / w_lat [K,6] f32, valid [K]
    bool, path_src [K] str, reason [K] str, checks [K,6] f32, check [K] str, abs_ok [K] bool, flags [K,11] f32,
    ctx (decoder.HumanContext), rng_hits (remapped A_p draws).
    """
    cfg = cfg or BankConfig()
    rng = bank_rng(token, cfg)
    ctx = D.HumanContext(tau_h, path_long, n_valid, v0, a0, None)
    if centerline is not None and needs_centerline(ctx, cfg):
        cl = centerline() if callable(centerline) else centerline
        ctx.centerline = None if cl is None else np.asarray(cl, np.float64).reshape(-1, 2)
    kw = dict(mode=cfg.mode, max_tries=cfg.max_tries, centerline_families=tuple(cfg.centerline_families))
    out = []
    for fam in cfg.layout:                       # same flow as decoder.sample_bank
        f = fam
        if f == "ignore_brake" and not ctx.decel_ok:
            f = "lconst"
        if f == "creep" and not ctx.creep_ok:
            f = "lconst"
        if f in ("lat", "combined") and not ctx.lat_ok:
            f = "lconst"
        r = D.sample_perturbation(f, ctx, rng, **kw)
        if not r["valid"] and f == "cv":
            r = D.sample_perturbation("hdrift", ctx, rng, mode=cfg.mode)
        elif not r["valid"] and f in ("lat", "combined", "ignore_brake", "creep"):
            r2 = D.sample_perturbation("lconst", ctx, rng, **kw)
            r2["reason"] = f"fallback_from_{f}:{r['reason']}" if r2["valid"] else r2["reason"]
            r = r2
        out.append(r)
    drafts = np.stack([r["draft"] for r in out]).astype(np.float32)
    valid = np.array([r["valid"] for r in out], bool)
    reason = [r["reason"] for r in out]
    vals, chk, abs_ok = draft_checks(ctx, drafts)
    for k in range(len(out)):
        if valid[k] and chk[k]:
            valid[k] = False
            reason[k] = f"post_check:{chk[k]}"
    flags = np.full((len(out), len(FLAG_COLS)), np.nan, np.float32)
    for k, r in enumerate(out):
        for j, c in enumerate(FLAG_COLS):
            if c in r["flags"]:
                flags[k, j] = r["flags"][c]
    return {
        "drafts": drafts,
        "family": np.array([r["code"] for r in out], np.int8),
        "slot": np.array([D.FAMILY[f] for f in cfg.layout], np.int8),
        "params": np.stack([r["params"] for r in out]).astype(np.float32),
        "z_lon": np.stack([r["z_lon"] for r in out]).astype(np.float32),
        "w_lat": np.stack([r["w_lat"] for r in out]).astype(np.float32),
        "valid": valid,
        "path_src": np.array([r["path_src"] for r in out]),
        "reason": np.array(reason),
        "checks": vals, "check": chk, "abs_ok": abs_ok, "flags": flags,
        "ctx": ctx, "rng_hits": int(getattr(rng, "hits", 0)),
    }


def bank_arrays(bank: Dict, token: str, log: str, split: str, cfg: BankConfig) -> Dict[str, np.ndarray]:
    """npz payload of one token (no object arrays)."""
    ctx = bank["ctx"]
    d = {k: bank[k] for k in ("drafts", "family", "params", "slot", "valid", "reason", "path_src", "z_lon", "w_lat",
                              "checks", "check", "abs_ok", "flags")}
    d.update(token=np.array(token), log=np.array(log), split=np.array(split),
             v0=np.float32(np.nan if ctx.v0 is None else ctx.v0), a0=np.float32(np.nan if ctx.a0 is None else ctx.a0),
             S8=np.float32(ctx.S8), S_avail=np.float32(ctx.S_avail), cfg_json=np.array(cfg.to_json()),
             cfg_hash=np.array(cfg.hash), version=np.array(VERSION), check_cols=np.array(CHECK_COLS),
             flag_cols=np.array(FLAG_COLS), param_cols=np.array(D.PARAM_COLS))
    for k in ("reason", "path_src", "check"):
        d[k] = np.asarray(d[k]).astype(str)
    return d


def save_bank(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    """Atomic write (tmp name does not end in .npz so directory scanners never see a partial file)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def load_bank(path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------------------------- inputs
def load_human(split: str, root: Path = HUMAN) -> Dict[str, np.ndarray]:
    z = np.load(Path(root) / f"{split}.npz", allow_pickle=False)
    d = {k: z[k] for k in ("tokens", "logs", "traj", "path", "n_reg", "v0", "a0", "frame_gap")}
    d["index"] = {t: i for i, t in enumerate(d["tokens"].tolist())}
    return d


def mc_ready(token: str, log: Optional[str], roots: Sequence[Path], min_age: float,
             now: Optional[float] = None) -> Tuple[str, Optional[str]]:
    """('ok', path) | ('no_mc', None) | ('mc_fresh', path) for the token's metric cache."""
    import score_trajectories as ST
    p = ST.locate_metric_cache(token, log or None, roots)
    if p is None:
        return "no_mc", None
    if min_age > 0 and (now or time.time()) - os.stat(p).st_mtime < min_age:
        return "mc_fresh", p
    return "ok", p


# ----------------------------------------------------------------------------------------------- generation
def _gen_one(task) -> Dict:
    """task = (token, log, split, tau_h, path_long, n_valid, v0, a0, mc_path, out_path, cfg_json) -> stats row."""
    token, log, split, tau_h, path_long, n_valid, v0, a0, mc_path, out_path, cfg_json = task
    t0 = time.time()
    st = {"token": token, "status": "built", "sec": 0.0, "sec_mc": 0.0, "centerline": False}
    try:
        import torch
        torch.set_num_threads(1)
        cfg = BankConfig.from_json(cfg_json)

        def _cl():
            import score_trajectories as ST
            ta = time.time()
            cl = centerline_n(ST.load_metric_cache(mc_path)) if mc_path else None
            st["sec_mc"] = time.time() - ta
            st["centerline"] = cl is not None
            return cl

        bank = make_bank(tau_h, token, path_long=path_long, n_valid=n_valid, v0=v0, a0=a0, centerline=_cl, cfg=cfg)
        save_bank(Path(out_path), bank_arrays(bank, token, log, split, cfg))
        st["n_valid"] = int(bank["valid"].sum())
        st["rng_hits"] = bank["rng_hits"]
    except Exception as e:  # never raises: the token is retried by the next pass
        st.update(status="error", err=(repr(e) + " | " + traceback.format_exc()[-600:])[:900])
    st["sec"] = time.time() - t0
    return st


def _config_guard(out_dir: Path, cfg: BankConfig, overwrite: bool) -> None:
    p = out_dir / "_config.json"
    if p.exists():
        old = json.loads(p.read_text())
        if old.get("cfg_hash") != cfg.hash and not overwrite:
            raise SystemExit(f"{out_dir} was generated with config {old.get('cfg_hash')} {old.get('cfg')}, "
                             f"now {cfg.hash} {cfg.to_json()}: refusing to mix (use another --out-dir or --overwrite)")
    p.write_text(json.dumps({"cfg_hash": cfg.hash, "cfg": json.loads(cfg.to_json()),
                             "written": time.strftime("%F %T")}, indent=1))


def generate(split: str, tokens: Optional[Sequence[str]], out_dir: Path, cfg: BankConfig, workers: int = 2,
             min_age: float = 120.0, roots: Optional[Sequence[Path]] = None, limit: int = 0,
             overwrite: bool = False, human_root: Path = HUMAN, log_every: int = 2000) -> Dict:
    """Draft npz for every requested token (default: all of human/<split>.npz) that has no npz yet, is not a
    frame-gap token and has a ready metric cache.  Returns counts per status + timing."""
    import score_trajectories as ST
    assert 1 <= workers <= MAX_WORKERS, f"at most {MAX_WORKERS} workers"
    roots = [Path(r) for r in (roots or ST.MC_ROOTS)]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _config_guard(out_dir, cfg, overwrite)
    H = load_human(split, human_root)
    toks = list(tokens) if tokens is not None else H["tokens"].tolist()
    have = {e.name[:-4] for e in os.scandir(out_dir) if e.name.endswith(".npz")}
    counts: Dict[str, int] = {}
    tasks = []
    now = time.time()
    for t in toks:
        i = H["index"].get(t)
        if i is None:
            st = "not_in_human"
        elif bool(H["frame_gap"][i]):
            st = "frame_gap"
        elif t in have and not overwrite:
            st = "exists"
        else:
            log = str(H["logs"][i])
            st, mcp = mc_ready(t, log, roots, min_age, now)
            if st == "ok":
                tasks.append((t, log, split, H["traj"][i], H["path"][i], int(H["n_reg"][i]), float(H["v0"][i]),
                              float(H["a0"][i]), mcp, str(out_dir / f"{t}.npz"), cfg.to_json()))
                st = "todo"
        counts[st] = counts.get(st, 0) + 1
    if limit:
        tasks = tasks[:limit]
    print(f"[generate {split}] {len(toks)} tokens {counts} -> building {len(tasks)} into {out_dir} "
          f"(cfg {cfg.hash}, workers {workers})", flush=True)
    t0 = time.time()
    rows = []
    if tasks:
        it = (map(_gen_one, tasks) if workers == 1 else None)
        pool = None
        if it is None:
            pool = Pool(workers, maxtasksperchild=2000)
            it = pool.imap_unordered(_gen_one, tasks, chunksize=8)
        try:
            for j, r in enumerate(it):
                rows.append(r)
                if (j + 1) % log_every == 0 or j + 1 == len(tasks):
                    el = time.time() - t0
                    nerr = sum(x["status"] == "error" for x in rows)
                    print(f"[generate {split}] {j + 1}/{len(tasks)} err {nerr} {el:.0f}s "
                          f"{(j + 1) / max(el, 1e-9):.1f} tok/s ETA {(len(tasks) - j - 1) / max((j + 1) / el, 1e-9) / 60:.1f} min",
                          flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    errs = [r for r in rows if r["status"] == "error"]
    for r in errs[:5]:
        print(f"[generate {split}] ERROR {r['token']}: {r['err']}", flush=True)
    sec = np.array([r["sec"] for r in rows]) if rows else np.zeros(0)
    return {"split": split, "counts": counts, "built": len(rows) - len(errs), "errors": len(errs),
            "error_tokens": [r["token"] for r in errs][:50], "wall_s": time.time() - t0,
            "sec_per_token": {"mean": float(sec.mean()) if len(sec) else None,
                              "p95": float(np.percentile(sec, 95)) if len(sec) else None},
            "centerline_loaded": int(sum(bool(r.get("centerline")) for r in rows))}


# ----------------------------------------------------------------------------------------------- scoring
def score(drafts_dir: Path, out: Path, workers: int = 2, tokens: Optional[Sequence[str]] = None,
          shard_size: int = 200, retry_errors: bool = True, logs: Optional[Dict[str, str]] = None) -> Dict:
    """Official labels of every drafted token (optionally restricted to ``tokens``) with score_trajectories.py
    (resumable shards next to ``out``; merged into ``out`` once every listed token is scored).  ``logs``: token ->
    log (default: read from each npz); used to locate the metric cache directly."""
    import pandas as pd
    import score_trajectories as ST
    assert 1 <= workers <= MAX_WORKERS, f"at most {MAX_WORKERS} workers"
    drafts_dir, out = Path(drafts_dir), Path(out)
    have = sorted(e.name[:-4] for e in os.scandir(drafts_dir) if e.name.endswith(".npz"))
    if tokens is not None:
        want = set(tokens)
        have = [t for t in have if t in want]
    log_list = []
    for t in have:  # log (the metric cache is then located directly)
        if logs is not None and t in logs:
            log_list.append(str(logs[t]))
            continue
        with np.load(drafts_dir / f"{t}.npz", allow_pickle=False) as z:
            log_list.append(str(z["log"]))
    tok_dir = out.parent / "_tokens"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tok_file = tok_dir / f"{out.stem}.parquet"
    pd.DataFrame({"token": have, "log": log_list}).to_parquet(tok_file, index=False)
    argv = ["--drafts", str(drafts_dir), "--tokens", str(tok_file), "--out", str(out), "--workers", str(workers),
            "--shard-size", str(shard_size)] + (["--retry-errors"] if retry_errors else [])
    t0 = time.time()
    ST.main(argv)
    meta = json.loads(Path(str(out) + ".meta.json").read_text())
    return {"n_tokens": len(have), "complete": bool(meta.get("complete")), "rows": meta.get("rows"),
            "run": meta.get("run"), "wall_s": time.time() - t0}


# ----------------------------------------------------------------------------------------------- analysis
def bank_table(drafts_dir: Path, tokens: Optional[Sequence[str]] = None):
    """One row per (token, k) from the npz files: slot / family names, valid, params, path_src, reason, checks."""
    import pandas as pd
    drafts_dir = Path(drafts_dir)
    if tokens is None:
        tokens = sorted(e.name[:-4] for e in os.scandir(drafts_dir) if e.name.endswith(".npz"))
    cols: Dict[str, list] = {c: [] for c in ("token", "k", "slot", "family", "valid", "reason", "path_src", "check",
                                            "abs_ok", "v0", "S8", *D.PARAM_COLS, *CHECK_COLS, "beta", "alpha")}
    ib, ia = FLAG_COLS.index("beta"), FLAG_COLS.index("alpha")
    for t in tokens:
        p = drafts_dir / f"{t}.npz"
        if not p.exists():
            continue
        z = load_bank(p)
        K = len(z["family"])
        cols["token"] += [t] * K
        cols["k"] += list(range(K))
        cols["slot"] += [D.FAMILY_NAME[int(f)] for f in z["slot"]]
        cols["family"] += [D.FAMILY_NAME[int(f)] for f in z["family"]]
        cols["valid"] += z["valid"].tolist()
        cols["reason"] += z["reason"].tolist()
        cols["path_src"] += z["path_src"].tolist()
        cols["check"] += z["check"].tolist()
        cols["abs_ok"] += z["abs_ok"].tolist()
        cols["v0"] += [float(z["v0"])] * K
        cols["S8"] += [float(z["S8"])] * K
        for j, c in enumerate(D.PARAM_COLS):
            cols[c] += z["params"][:, j].tolist()
        for j, c in enumerate(CHECK_COLS):
            cols[c] += z["checks"][:, j].tolist()
        cols["beta"] += z["flags"][:, ib].tolist()
        cols["alpha"] += z["flags"][:, ia].tolist()
    return pd.DataFrame(cols)


def _q(x, qs=(1, 5, 50, 95, 99)) -> Dict[str, float]:
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {}
    return {f"p{q}": round(float(np.percentile(x, q)), 4) for q in qs}


def _rates(df) -> Dict[str, float]:
    n = int(len(df))
    if n == 0:
        return {"n": 0}
    r = {"n": n,
         "fail": float(df.fail.mean()), "fail_ttc": float(df.fail_ttc.mean()),
         "nc_fail": float((df.nc < 1).mean()), "nc0": float((df.nc == 0).mean()),
         "dac_fail": float((df.dac < 1).mean()), "ddc_fail": float((df.ddc < 1).mean()),
         "ttc_fail": float((df.ttc < 1).mean()), "comfort_fail": float((df.comfort < 1).mean()),
         "ep_mean": float(df.ep.mean()), "pdms_mean": float(df.pdms.mean())}
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}


def summarize(drafts_dir: Path, scores, tokens: Optional[Sequence[str]] = None) -> Dict:
    """Family-wise official failure rates of a scored bank (scores: DataFrame or parquet path)."""
    import pandas as pd
    S = pd.read_parquet(scores) if not isinstance(scores, pd.DataFrame) else scores
    err_tok = sorted(set(S.loc[S.k < 0, "token"])) if "k" in S else []
    S = S[S.k >= 0]
    if tokens is not None:
        S = S[S.token.isin(set(tokens))]
    B = bank_table(drafts_dir, sorted(set(S.token)))
    keep = ["token", "k", *SCORE_COLS, "pdm_progress_eff", "raw_progress", "sec_load", "sec_score"]
    M = B.merge(S[[c for c in keep if c in S.columns]], on=["token", "k"], how="inner")
    M["fail"] = (M.nc < 1) | (M.dac < 1) | (M.ddc < 1)
    M["fail_ttc"] = M.fail | (M.ttc < 1)
    V = M[M.valid]
    P = V[V.family != "identity"]
    out = {"n_tokens": int(M.token.nunique()), "n_rows": int(len(M)), "n_error_tokens": len(err_tok),
           "fail_definition": "fail = NC<1 | DAC<1 | DDC<1 (IMPL_SPEC 3.7 gate label); fail_ttc adds TTC<1",
           "overall": {"all_rows": _rates(M), "valid_rows": _rates(V), "valid_perturbed_rows": _rates(P)},
           "valid_rate": round(float(M.valid.mean()), 4),
           "per_family": {f: _rates(g) for f, g in V.groupby("family")},
           "per_slot": {f"{k:02d}_{g.slot.iloc[0]}": _rates(g) for k, g in V.groupby("k")},
           "valid_rate_per_slot": {f"{k:02d}_{g.slot.iloc[0]}": round(float(g.valid.mean()), 4)
                                   for k, g in M.groupby("k")},
           "family_count_valid": {f: int(n) for f, n in V.family.value_counts().items()},
           "invalid_reasons": {str(r): int(n) for r, n in M.loc[~M.valid, "reason"].value_counts().head(15).items()},
           "fallback_reasons": {str(r): int(n) for r, n in
                                M.loc[M.valid & M.reason.str.startswith("fallback"), "reason"]
                                .str.split(":").str[0].value_counts().items()},
           "path_src": {s: _rates(g) for s, g in V.groupby("path_src")}}
    # identity sanity (human trajectories): official failures should be ~0
    I = M[M.family == "identity"]
    out["identity_fail_tokens"] = sorted(I.loc[I.fail, "token"].tolist())[:50]
    # L-const / combined: failure vs realised A_p and t_on
    L = V[V.family.isin(["lconst", "combined"])]
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    if len(L):
        cut = pd.cut(L.A_p, bins, right=False)
        out["lconst_combined_by_Ap"] = {str(b): _rates(g) for b, g in L.groupby(cut, observed=True)}
        out["lconst_combined_by_ton"] = {f"{t:.1f}": _rates(g) for t, g in L.groupby("t_on")}
        out["lconst_Ap_realised"] = _q(L.A_p, (0, 5, 50, 95, 100))
        out["lconst_beta_lt1"] = round(float((L.beta < 1 - 1e-6).mean()), 4)
    Lt = V[V.family.isin(["lat", "combined"])]
    if len(Lt):
        cut = pd.cut(Lt.D_p.abs(), [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0], right=False)
        out["lat_by_absDp"] = {str(b): _rates(g) for b, g in Lt.groupby(cut, observed=True)}
    # v0 strata (failure of speed-ups depends on speed)
    cut = pd.cut(V.v0, [0, 2, 5, 10, 15, 40], right=False)
    out["perturbed_by_v0"] = {str(b): _rates(g[g.family != "identity"]) for b, g in V.groupby(cut, observed=True)}
    # t0 continuity / kinematics of valid drafts
    out["first_dv_by_family"] = {f: _q(g.first_dv) for f, g in V.groupby("family")}
    out["abs_limits_ok_by_family"] = {f: round(float(g.abs_ok.mean()), 4) for f, g in V.groupby("family")}
    out["kinematics_valid"] = {c: _q(V[c]) for c in ("acc_min", "acc_max", "lat_max", "kappa_max")}
    # token level
    T = V.groupby("token").agg(n_fail=("fail", "sum"), n=("fail", "size"))
    out["token_level"] = {"frac_tokens_with_fail": round(float((T.n_fail > 0).mean()), 4),
                          "mean_fail_per_token": round(float(T.n_fail.mean()), 3),
                          "mean_valid_per_token": round(float(T.n.mean()), 3)}
    if "sec_score" in M:
        tt = M.drop_duplicates("token")
        out["scoring_sec_per_token"] = {"load": _q(tt.sec_load, (50, 90)), "score": _q(tt.sec_score, (50, 90))}
    return out


# ----------------------------------------------------------------------------------------------- pilot / run
def pilot_tokens(n: int, seed: int, roots, min_age: float, path: Path) -> List[str]:
    """Fixed pilot token list (drawn once): n random non-gap TRAIN tokens with a ready metric cache."""
    import pandas as pd
    if path.exists():
        return pd.read_parquet(path).token.tolist()
    H = load_human("train")
    now = time.time()
    ok = [i for i, t in enumerate(H["tokens"].tolist())
          if not H["frame_gap"][i] and mc_ready(t, str(H["logs"][i]), roots, min_age, now)[0] == "ok"]
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(ok, min(n, len(ok)), replace=False))
    df = pd.DataFrame({"token": H["tokens"][pick], "log": H["logs"][pick]})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    (path.with_suffix(".json")).write_text(json.dumps({"n": int(len(df)), "seed": seed, "pool_ready": len(ok),
                                                       "pool_train": int(len(H["tokens"])),
                                                       "drawn": time.strftime("%F %T")}, indent=1))
    return df.token.tolist()


def cmd_pilot(a):
    import score_trajectories as ST
    cfg = _cfg_from_args(a)
    roots = [Path(r) for r in ST.MC_ROOTS]
    toks = pilot_tokens(a.n, a.seed, roots, a.min_age, DRAFTS / "_pilot" / "tokens.parquet")
    ddir = DRAFTS / "_pilot" / a.tag
    out = SCORES / "_pilot" / f"{a.tag}.parquet"
    g = generate("train", toks, ddir, cfg, a.workers, a.min_age, roots, overwrite=a.overwrite)
    s = score(ddir, out, a.workers, toks, shard_size=50)
    res = {"tag": a.tag, "cfg": json.loads(cfg.to_json()), "cfg_hash": cfg.hash, "generate": g, "score": s,
           "summary": summarize(ddir, out, toks)}
    REPORT.mkdir(parents=True, exist_ok=True)
    p = REPORT / f"draft_bank_pilot_{a.tag}.json"
    p.write_text(json.dumps(res, indent=1))
    o = res["summary"]["overall"]
    print(json.dumps({"tag": a.tag, "valid_rows": o["valid_rows"], "valid_perturbed_rows": o["valid_perturbed_rows"]},
                     indent=1))
    print(f"-> {p}")


def cmd_compare(a):
    res = {}
    for t in a.tags:
        r = json.loads((REPORT / f"draft_bank_pilot_{t}.json").read_text())
        s = r["summary"]
        res[t] = {"cfg": r["cfg"], "cfg_hash": r["cfg_hash"], "n_tokens": s["n_tokens"],
                  "overall": s["overall"], "valid_rate": s["valid_rate"],
                  "per_family_fail": {f: v.get("fail") for f, v in s["per_family"].items()},
                  "per_family_fail_ttc": {f: v.get("fail_ttc") for f, v in s["per_family"].items()},
                  "per_family_n": {f: v.get("n") for f, v in s["per_family"].items()},
                  "per_family_nc_fail": {f: v.get("nc_fail") for f, v in s["per_family"].items()},
                  "lconst_combined_by_Ap": {b: (v.get("n"), v.get("fail"), v.get("nc_fail"))
                                            for b, v in s.get("lconst_combined_by_Ap", {}).items()},
                  "path_src": {k: (v.get("n"), v.get("fail"), v.get("nc_fail")) for k, v in s["path_src"].items()},
                  "invalid_reasons": s["invalid_reasons"],
                  "token_level": s["token_level"], "details": f"draft_bank_pilot_{t}.json"}
    res["chosen"] = a.chosen or a.tags[-1]
    res["target"] = "official bank failure rate (NC|DAC|DDC < 1) over valid drafts in [0.20, 0.35]"
    if a.note:
        res["note"] = a.note
    p = REPORT / "draft_bank_pilot.json"
    p.write_text(json.dumps(res, indent=1))
    print(json.dumps({t: (v["overall"]["valid_rows"]["fail"] if isinstance(v, dict) and "overall" in v else v)
                      for t, v in res.items()}, indent=1))
    print(f"-> {p}")


def _cfg_from_args(a) -> BankConfig:
    kw = {}
    if getattr(a, "dp", None):
        kw["lat_dp"] = tuple(a.dp)
    if getattr(a, "centerline_families", ""):
        kw["centerline_families"] = tuple(a.centerline_families.split(","))
    return BankConfig(lconst_ap=tuple(a.ap) if a.ap else TUNED_AP, **kw)


def cmd_generate(a):
    cfg = _cfg_from_args(a)
    out_dir = Path(a.out_dir) if a.out_dir else DRAFTS / a.split
    toks = None
    if a.tokens:
        import pandas as pd
        toks = pd.read_parquet(a.tokens).token.astype(str).tolist()
    r = generate(a.split, toks, out_dir, cfg, a.workers, a.min_age, limit=a.limit, overwrite=a.overwrite)
    print(json.dumps(r, indent=1))


def cmd_score(a):
    ddir = Path(a.drafts_dir) if a.drafts_dir else DRAFTS / a.split
    out = Path(a.out) if a.out else SCORES / f"{a.split}.parquet"
    print(json.dumps(score(ddir, out, a.workers), indent=1))


def cmd_summarize(a):
    r = summarize(Path(a.drafts_dir), a.scores)
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(json.dumps(r["overall"], indent=1))


MC_BUILD_LOG = DATA / "metric_cache" / "logs" / "build.log"
MC_MANIFEST = DATA / "metric_cache" / "manifest.parquet"


def mc_build_done(idle_s: float = 3600.0) -> bool:
    """The stage-T metric-cache job has ended: its manifest exists, or its log has been idle for idle_s."""
    if MC_MANIFEST.exists():
        return True
    try:
        return time.time() - MC_BUILD_LOG.stat().st_mtime > idle_s
    except OSError:
        return True


def cmd_run(a):
    """Passes of generate -> score for each split until every non-gap token is drafted + scored, or the metric-cache
    job has ended and a pass finds nothing new to draft (tokens that never got a cache stay listed as no_mc in
    drafts/logs/run_<split>.jsonl), or the time limit.  The summary JSON is (re)written when a split finishes."""
    cfg = _cfg_from_args(a)
    splits = a.splits.split(",")
    deadline = time.time() + a.follow_max_h * 3600
    npass = 0
    while True:
        npass += 1
        done_all = True
        mc_done = mc_build_done()
        for split in splits:
            ddir, out = DRAFTS / split, SCORES / f"{split}.parquet"
            print(f"===== pass {npass} split {split} {time.strftime('%F %T')} (metric-cache job done: {mc_done})",
                  flush=True)
            g = generate(split, None, ddir, cfg, a.workers, a.min_age)
            H = load_human(split)
            n_target = int((~H["frame_gap"]).sum())
            n_drafted = sum(1 for e in os.scandir(ddir) if e.name.endswith(".npz"))
            s = score(ddir, out, a.workers, logs=dict(zip(H["tokens"].tolist(), H["logs"].tolist())))
            status = {"split": split, "pass": npass, "time": time.strftime("%F %T"), "n_target": n_target,
                      "n_drafted": n_drafted, "mc_build_done": mc_done, "generate": g, "score": s}
            (DRAFTS / "logs").mkdir(parents=True, exist_ok=True)
            with open(DRAFTS / "logs" / f"run_{split}.jsonl", "a") as f:
                f.write(json.dumps(status) + "\n")
            print(f"[run {split}] drafted {n_drafted}/{n_target}, scored complete={s['complete']} rows={s['rows']}",
                  flush=True)
            nothing_new = g["built"] == 0 and g["counts"].get("mc_fresh", 0) == 0
            finished = bool(s["complete"] and (n_drafted >= n_target or (mc_done and nothing_new)))
            last = time.time() + a.follow_interval > deadline
            if s["complete"] and (finished or last):
                r = summarize(ddir, out)
                r.update(cfg=json.loads(cfg.to_json()), cfg_hash=cfg.hash, n_target=n_target,
                         n_drafted=n_drafted, finished=finished, written=time.strftime("%F %T"),
                         not_drafted=g["counts"])
                sp = REPORT / f"draft_bank_{split}.json"
                sp.write_text(json.dumps(r, indent=1))
                print(f"[run {split}] summary -> {sp}", flush=True)
            done_all &= finished
        if done_all:
            print("all splits drafted and scored", flush=True)
            break
        if time.time() + a.follow_interval > deadline:
            print("time limit reached; re-run the same command to resume", flush=True)
            break
        print(f"sleeping {a.follow_interval:.0f}s", flush=True)
        time.sleep(a.follow_interval)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, split=True):
        if split:
            p.add_argument("--split", default="train")
        p.add_argument("--workers", type=int, default=2)
        p.add_argument("--min-age", type=float, default=120.0, help="metric caches written < this many s ago wait")
        p.add_argument("--ap", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                       help=f"L-const / combined A_p range (default {TUNED_AP})")
        p.add_argument("--dp", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                       help=f"lat / combined |D_p| range (default {DECODER_DP}; exploration option)")
        p.add_argument("--centerline-families", default="",
                       help="comma list of families allowed to continue on the route centerline (default creep; "
                            "exploration option)")
        p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("pilot")
    common(p, split=False)
    p.add_argument("--tag", required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_pilot)
    p = sub.add_parser("compare")
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--chosen", default="")
    p.add_argument("--note", default="")
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("generate")
    common(p)
    p.add_argument("--tokens", default="")
    p.add_argument("--out-dir", default="")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=cmd_generate)
    p = sub.add_parser("score")
    p.add_argument("--split", default="train")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--drafts-dir", default="")
    p.add_argument("--out", default="")
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser("summarize")
    p.add_argument("--drafts-dir", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_summarize)
    p = sub.add_parser("run")
    common(p, split=False)
    p.add_argument("--splits", default="train,dev")
    p.add_argument("--follow-interval", type=float, default=900.0)
    p.add_argument("--follow-max-h", type=float, default=12.0)
    p.set_defaults(fn=cmd_run)
    a = ap.parse_args(argv)
    if getattr(a, "workers", 1) > MAX_WORKERS:
        raise SystemExit(f"at most {MAX_WORKERS} workers (shared machine)")
    a.fn(a)


if __name__ == "__main__":
    main()
