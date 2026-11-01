"""Raw-anchor sampling for the r34-free teacher CK (CK2; user decision 2026-10-08).

The teacher CK is trained from scratch on the 256 fixed raw anchors (trajectory_anchors_256.npy, [256, 8, 3]
x fwd / y left / heading, ego frame at t = 0), with official per-trajectory labels.  Per token and epoch it sees
K = 32 of them:

  near   n_near = 8   the 8 anchors nearest to the GT (human) trajectory              (deterministic, pi = 1)
  mid    n_mid  = 8   uniform without replacement from GT-distance ranks 8 .. 71      (pi = n_mid / mid_pool = 1/8)
                      (the next mid_pool = 64)
  strat  n_strat = 16 from the far region, ranks 72 .. 255 (184 anchors), stratified by the official label:
                      8 pass + 8 fail, uniform without replacement inside each class  (pi = 8 / |P_far|, 8 / |F_far|)
                      pass = NC == DAC == TTC == C == 1 (EP is not part of it)

  Strata are disjoint by GT-distance rank: the 56 mid-pool anchors not drawn for 'mid' are NOT eligible for 'strat'.
  That makes every inclusion probability exact and closed-form (stratified SRS), so the Horvitz-Thompson weights
  w = 1 / pi of one token always sum to exactly 256.

  Fallback (one class short in the far region): all of the short class is taken and the rest of the 16 is filled
  from the other class (pi recomputed accordingly; Sum w is still 256).  Recorded per sample in info['fallback']
  ('none' | 'pass_short' | 'fail_short').  Measured on 2000 random tokens: |P_far| < 8 for 17 % (navtrain_val) /
  20 % (navtrain_train) of tokens; |F_far| < 8 never occurred.

GT distance (= v2's WTA anchor matching, modules/anchor_planner.anchor_plan_losses):
    dist_k = || anchor_k.reshape(24) - gt.reshape(24) ||_2     over (x, y, heading) of all 8 poses, heading in rad,
                                                               no angle wrapping, no per-dim weights
  ranks = argsort(dist, stable) so a tie goes to the lower anchor index (torch argmin keeps the first minimum):
  near[0] is v2's WTA winner.  Computed in float64 here (v2: float32 torch.norm); only near-exact ties can differ.

GT trajectory (training-time sampling only; never a model input):
  navtrain: packed/<split>/gt_traj.npy (row = packed/<split>/tokens.parquet).  Written by tools/ck/data/pack_v2.py from
  the v2 dump (tools/ck/data/dump_v2.py FeatDS: scene.get_future_trajectory(8).poses), the same call
  ParaSSRTargetBuilder uses for targets['trajectory'] (the tensor v2's anchor_plan_losses matches against).  Checked
  bit-exact on the 2 real-batch fixture tokens (rows 0, 40000).  The e2e / online path has targets['trajectory'].

Determinism: one numpy PCG64 stream per (seed, epoch, token), seeded from blake2b("ck2.anchor_sampler|seed|epoch|
token") -- independent of process, worker, call order and PYTHONHASHSEED.  The output slot order (shuffle=True,
default) uses a second stream ("...|perm"), so the drawn set does not depend on shuffle.

Label source (LabelSource): [5, 256] per token, key order NC, DAC, EP, TTC, C (= CK_KEYS = pdm_score_256 axis 1):
  preferred  CK_DATA/raw256/pdm_score_256_officialEP.npy (+ .tokens.json): official score_token of all 256 raw anchors
             (NC / DAC / TTC / C checked identical to pdm_score_256 on every navtrain token, EP official).
  fallback   v2_planning_data/planning_vb/pdm_score_256.npy (WoTE labels): NC / DAC / TTC / C used, EP set to NaN
             (WoTE's EP is a batch artifact).  The choice is logged (WARNING for the fallback) and kept in .kind.

Prior correction.  Unweighted training sees the sampled prior (the near + strat groups push the pass rate far above
the raw-anchor rate of ~22 %), while the CK is deployed on v2 r34's top-16 executed candidates (~88-93 % pass).
sample() returns pi and w = 1/pi per drawn anchor; split_priors() gives, per key (CK_KEYS + 'pass'):
  deploy      mean official label of v2 r34's executed candidates (labels/<split>/cand/labels.npy, all ok cands)
  sampled     expected label mean of what the trainer sees unweighted:  Sum_t Sum_i pi_ti y_ti / Sum_t Sum_i pi_ti
  population  uniform over the 256 anchors (what a w-weighted loss estimates):  Sum_t Sum_i y_ti / (256 N)
  offset_*    logit(deploy) - logit(*): add to the CK logit at deployment (prior-shift correction)
  offset_sampled_to_population  logit(population) - logit(sampled): undoes only the sampling design (unweighted
              training -> the uniform raw-anchor population, i.e. what the HT-weighted loss estimates directly)
  lsw_*       (w_pos, w_neg) = (deploy / p, (1 - deploy) / (1 - p)): label-shift importance weights for the
              positive / negative BCE terms (soft labels: y w_pos log s + (1 - y) w_neg log(1 - s))
  EP is continuous: its 'prior' is the mean EP and its offset is only indicative.  NC has 0.5 values (soft label).
  Caveat [estimate]: the offsets are global label-shift corrections.  The sampling bias is label shift only inside
  the far strata (per-token pi_pass / pi_fail); near / mid select on the GT distance.  population -> deploy (raw
  anchors -> v2 r34's top-16) is mostly a covariate shift (better trajectories), which a calibrated P(y | x) already
  handles: offset_population on top of an HT-weighted model can double count it.  Which correction to use is a
  training decision.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np

from . import constants as C

log = logging.getLogger(__name__)

N_ANCHORS = 256
ANCHORS_FILE = Path(C.ANCHORS)          # byte-identical copy: v2_planning_data/planning_vb/trajectory_anchors_256.npy
ANCHORS_SHA16 = "6bdf549edc25a959"      # sha256(float32 [256, 8, 3] bytes)[:16], measured 2026-10-07
OFFICIAL_LABELS = Path(C.CK_DATA) / "raw256" / "pdm_score_256_officialEP"
WOTE_LABELS = Path("/workspace/yongjae/ssd/yongjae_refiner/v2_planning_data/planning_vb/pdm_score_256")

LABEL_KEYS = tuple(C.CK_KEYS)           # nc, dac, ep, ttc, comfort: the [5, 256] axis-0 order
EP = LABEL_KEYS.index("ep")
PASS_KEYS = ("nc", "dac", "ttc", "comfort")
PASS_IDX = tuple(LABEL_KEYS.index(k) for k in PASS_KEYS)
PRIOR_KEYS = LABEL_KEYS + ("pass",)

GROUP_NEAR, GROUP_MID, GROUP_STRAT = 0, 1, 2
GROUP_NAMES = ("near", "mid", "strat")
# per-anchor design strata (design()['stratum']): near, mid pool, far pass, far fail
STRATUM_NAMES = ("near", "mid", "far_pass", "far_fail")
FALLBACKS = ("none", "pass_short", "fail_short")
_LOGIT_EPS = 1e-6


# ----------------------------------------------------------------------------------------------- basics
def load_anchors(path: Union[str, Path] = ANCHORS_FILE, check_sha: bool = True) -> np.ndarray:
    a = np.ascontiguousarray(np.load(path), dtype=np.float32)
    if a.shape != (N_ANCHORS, C.N_POSE, 3):
        raise ValueError(f"anchors must be [{N_ANCHORS}, {C.N_POSE}, 3], got {a.shape}")
    if check_sha:
        sha = hashlib.sha256(a.tobytes()).hexdigest()[:16]
        if sha != ANCHORS_SHA16:
            raise ValueError(f"anchors {path}: sha16 {sha} != {ANCHORS_SHA16}")
    return a


def gt_distance(anchors: np.ndarray, gt_traj: np.ndarray) -> np.ndarray:
    """v2 WTA metric: L2 over the flattened [8 x (x, y, heading)] difference -> [256] float64."""
    a = np.asarray(anchors, np.float64).reshape(len(anchors), -1)
    g = np.asarray(gt_traj, np.float64).reshape(1, -1)
    if g.shape[1] != a.shape[1]:
        raise ValueError(f"gt_traj {np.shape(gt_traj)} does not match anchors {np.shape(anchors)}")
    return np.sqrt(((a - g) ** 2).sum(1))


def pass_mask(labels: np.ndarray) -> np.ndarray:
    """labels [5, N] (LABEL_KEYS) -> bool [N]: NC == DAC == TTC == C == 1.  Raises on non-finite pass keys."""
    lab = np.asarray(labels, np.float32)
    sub = lab[list(PASS_IDX)]
    if not np.isfinite(sub).all():
        raise ValueError("labels: non-finite NC / DAC / TTC / C")
    return (sub >= 1.0 - 1e-6).all(0)


def token_rng(token: str, epoch: int, seed: int, stream: str = "") -> np.random.Generator:
    """PCG64 keyed by (seed, epoch, token[, stream]); stream '' = the draw stream (unchanged since v1), 'perm' = the
    output slot permutation (separate, so shuffling never changes which anchors are drawn)."""
    key = f"ck2.anchor_sampler|{int(seed)}|{int(epoch)}|{token}" + (f"|{stream}" if stream else "")
    h = hashlib.blake2b(key.encode(), digest_size=8)
    return np.random.Generator(np.random.PCG64(int.from_bytes(h.digest(), "little")))


def logit(p: float, eps: float = _LOGIT_EPS) -> float:
    if p is None or not np.isfinite(p):
        return float("nan")
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p) - math.log1p(-p)


def log_odds_offset(p_target: float, p_train: float) -> float:
    """Additive logit correction: model trained at prior p_train, deployed at prior p_target."""
    return logit(p_target) - logit(p_train)


def label_shift_weights(p_target: float, p_train: float, eps: float = _LOGIT_EPS):
    """(w_pos, w_neg) importance weights of the positive / negative BCE terms (prior p_train -> p_target)."""
    if not (np.isfinite(p_target) and np.isfinite(p_train)):
        return float("nan"), float("nan")
    pt = min(max(float(p_train), eps), 1.0 - eps)
    return float(p_target) / pt, (1.0 - float(p_target)) / (1.0 - pt)


# ----------------------------------------------------------------------------------------------- sampler
class AnchorSampler:
    """Per-token K-anchor sampler (near / mid / strat) with exact inclusion probabilities.

    sample(token, gt_traj [8, 3], labels [5, 256] or None (-> label_source), epoch, shuffle=True) -> dict
      idx       int64 [K]   anchor indices.  shuffle=True (default, for training): a deterministic random slot order
                            (own PCG64 stream 'perm' of (seed, epoch, token)), so the slot position carries no GT-rank
                            or pass / fail information.  shuffle=False (analysis, label_variants): canonical order
                            near (rank order; idx[0] = v2 WTA winner), mid (rank order), strat (pass then fail, each
                            in rank order) -- that order encodes GT rank and the label, never feed it to an
                            order-aware model.  The drawn set, pi and w are identical either way; every per-slot
                            array below is permuted together.
      group     int8  [K]   GROUP_NEAR / GROUP_MID / GROUP_STRAT (names GROUP_NAMES)
      pass_mask bool  [K]   NC == DAC == TTC == C == 1
      pi        f64   [K]   inclusion probability of the anchor under the design (given GT and labels)
      w         f32   [K]   Horvitz-Thompson weight 1 / pi; Sum w == 256 for every token
      rank      int16 [K]   GT-distance rank (0 = nearest)
      dist      f32   [K]   GT distance (v2 WTA metric)
      info      dict        n_pass_far, n_fail_far, n_pass_taken, n_fail_taken, fallback (FALLBACKS)
    group / pass_mask / pi / w / rank / dist / info are derived from the GT and the labels: training-time metadata
    (loss weights, prior correction, logging), never model inputs.
    """

    def __init__(self, anchors: Union[np.ndarray, str, Path, None] = None, label_source: Optional["LabelSource"] = None,
                 K: int = 32, n_near: int = 8, n_mid: int = 8, mid_pool: int = 64, n_strat: int = 16, seed: int = 0):
        a = load_anchors() if anchors is None else (
            load_anchors(anchors) if isinstance(anchors, (str, Path)) else np.ascontiguousarray(anchors, np.float32))
        if a.ndim != 3 or a.shape[2] != 3:
            raise ValueError(f"anchors must be [N, T, 3], got {a.shape}")
        n = a.shape[0]
        if K != n_near + n_mid + n_strat:
            raise ValueError(f"K={K} != n_near + n_mid + n_strat = {n_near + n_mid + n_strat}")
        if not (n_near >= 0 and 1 <= n_mid <= mid_pool and n_strat >= 1):
            raise ValueError(f"need n_near >= 0, 1 <= n_mid <= mid_pool, n_strat >= 1 "
                             f"(got {n_near}, {n_mid}, {mid_pool}, {n_strat})")
        if n_near + mid_pool + n_strat > n:
            raise ValueError(f"far region ({n - n_near - mid_pool}) smaller than n_strat={n_strat}")
        self.anchors = a
        self.n_anchors = n
        self.label_source = label_source
        self.K, self.n_near, self.n_mid, self.mid_pool, self.n_strat = int(K), int(n_near), int(n_mid), int(
            mid_pool), int(n_strat)
        self.n_strat_pass = self.n_strat // 2
        self.n_strat_fail = self.n_strat - self.n_strat_pass
        self.seed = int(seed)

    def config(self) -> Dict:
        return {"K": self.K, "n_near": self.n_near, "n_mid": self.n_mid, "mid_pool": self.mid_pool,
                "n_strat": self.n_strat, "n_strat_pass": self.n_strat_pass, "n_strat_fail": self.n_strat_fail,
                "seed": self.seed, "n_anchors": self.n_anchors, "strat_pool": "disjoint (ranks >= n_near + mid_pool)",
                "anchors_sha16": hashlib.sha256(self.anchors.tobytes()).hexdigest()[:16],
                "label_source": self.label_source.describe() if self.label_source is not None else None}

    # -------------------------------------------------------------------------------------------- design
    def _labels(self, token: str, labels) -> np.ndarray:
        if labels is None:
            if self.label_source is None:
                raise ValueError("no labels given and no label_source")
            labels = self.label_source.get(token)
            if labels is None:
                raise KeyError(f"token {token}: no labels in {self.label_source.describe()['path']}")
        lab = np.asarray(labels, np.float32)
        if lab.shape != (len(LABEL_KEYS), self.n_anchors):
            raise ValueError(f"labels must be [{len(LABEL_KEYS)}, {self.n_anchors}], got {lab.shape}")
        return lab

    def _far_takes(self, n_p: int, n_f: int):
        tp, tf = min(self.n_strat_pass, n_p), min(self.n_strat_fail, n_f)
        if tp < self.n_strat_pass:
            tf = min(n_f, self.n_strat - tp)
            fb = "pass_short"
        elif tf < self.n_strat_fail:
            tp = min(n_p, self.n_strat - tf)
            fb = "fail_short"
        else:
            fb = "none"
        return tp, tf, fb

    def design(self, gt_traj: np.ndarray, labels: np.ndarray) -> Dict:
        """Deterministic part of the design (no RNG): ranks, strata and the exact inclusion probability of every
        anchor.  gt_traj [8, 3], labels [5, 256] -> dict(order [256] anchor ids by GT distance, dist [256],
        rank [256], pass [256], stratum int8 [256] (STRATUM_NAMES), pi [256], n_pass_far, n_fail_far,
        n_pass_taken, n_fail_taken, fallback)."""
        g = np.asarray(gt_traj, np.float64)
        if not np.isfinite(g).all():
            raise ValueError("gt_traj is not finite")
        dist = gt_distance(self.anchors, g)
        order = np.argsort(dist, kind="stable")
        rank = np.empty(self.n_anchors, np.int64)
        rank[order] = np.arange(self.n_anchors)
        ok = pass_mask(labels)
        stratum = np.empty(self.n_anchors, np.int8)
        stratum[rank < self.n_near] = 0
        stratum[(rank >= self.n_near) & (rank < self.n_near + self.mid_pool)] = 1
        far = rank >= self.n_near + self.mid_pool
        stratum[far & ok] = 2
        stratum[far & ~ok] = 3
        n_p, n_f = int((stratum == 2).sum()), int((stratum == 3).sum())
        tp, tf, fb = self._far_takes(n_p, n_f)
        pi = np.zeros(self.n_anchors, np.float64)
        pi[stratum == 0] = 1.0
        pi[stratum == 1] = self.n_mid / self.mid_pool
        if n_p:
            pi[stratum == 2] = tp / n_p
        if n_f:
            pi[stratum == 3] = tf / n_f
        return {"order": order, "dist": dist, "rank": rank, "pass": ok, "stratum": stratum, "pi": pi,
                "n_pass_far": n_p, "n_fail_far": n_f, "n_pass_taken": tp, "n_fail_taken": tf, "fallback": fb}

    # -------------------------------------------------------------------------------------------- sample
    def sample(self, token: str, gt_traj: np.ndarray, labels: Optional[np.ndarray] = None, epoch: int = 0,
               shuffle: bool = True) -> Dict:
        lab = self._labels(token, labels)
        d = self.design(gt_traj, lab)
        rng = token_rng(token, epoch, self.seed)
        order, stratum = d["order"], d["stratum"]

        def draw(ids: np.ndarray, n: int) -> np.ndarray:      # ids in rank order -> n of them, kept in rank order
            if n >= len(ids):
                return ids
            return ids[np.sort(rng.permutation(len(ids))[:n])]

        near = order[:self.n_near]
        mid = draw(order[self.n_near:self.n_near + self.mid_pool], self.n_mid)
        far = order[self.n_near + self.mid_pool:]
        sp = draw(far[stratum[far] == 2], d["n_pass_taken"])
        sf = draw(far[stratum[far] == 3], d["n_fail_taken"])
        idx = np.concatenate([near, mid, sp, sf]).astype(np.int64)
        group = np.concatenate([np.full(len(near), GROUP_NEAR), np.full(len(mid), GROUP_MID),
                                np.full(len(sp) + len(sf), GROUP_STRAT)]).astype(np.int8)
        assert len(idx) == self.K and len(np.unique(idx)) == self.K, (len(idx), d["fallback"])
        if shuffle:                                           # slot order must not encode GT rank / label
            p = token_rng(token, epoch, self.seed, "perm").permutation(self.K)
            idx, group = idx[p], group[p]
        pi = d["pi"][idx]
        return {"idx": idx, "group": group, "pass_mask": d["pass"][idx], "pi": pi, "w": (1.0 / pi).astype(np.float32),
                "rank": d["rank"][idx].astype(np.int16), "dist": d["dist"][idx].astype(np.float32),
                "info": {k: d[k] for k in ("n_pass_far", "n_fail_far", "n_pass_taken", "n_fail_taken", "fallback")}}


# ----------------------------------------------------------------------------------------------- data sources
class LabelSource:
    """token -> [5, 256] float32 labels (LABEL_KEYS order).  Prefers the official-EP raw256 file; otherwise the WoTE
    pdm_score_256 with EP = NaN.  Opens lazily (mmap), so it can be pickled into dataloader workers."""

    def __init__(self, prefer_official: bool = True, official: Union[str, Path] = OFFICIAL_LABELS,
                 fallback: Union[str, Path] = WOTE_LABELS):
        official, fallback = str(official), str(fallback)
        has_official = Path(official + ".npy").is_file() and Path(official + ".tokens.json").is_file()
        if prefer_official and has_official:
            self.path, self.kind, self.ep_valid = official, "official_ep", True
            log.info("LabelSource: official per-trajectory labels %s.npy (NC/DAC/EP/TTC/C)", official)
        else:
            if not (Path(fallback + ".npy").is_file() and Path(fallback + ".tokens.json").is_file()):
                raise FileNotFoundError(f"no label file: {official}.npy / {fallback}.npy")
            self.path, self.kind, self.ep_valid = fallback, "wote_ep_masked", False
            why = "official file missing" if not has_official else "prefer_official=False"
            log.warning("LabelSource: FALLBACK to WoTE %s.npy (%s): NC/DAC/TTC/C only, EP masked to NaN", fallback, why)
        self._scores = None
        self._index = None

    def __getstate__(self):
        s = dict(self.__dict__)
        s["_scores"] = None
        s["_index"] = None
        return s

    def _open(self):
        if self._scores is None:
            self._scores = np.load(self.path + ".npy", mmap_mode="r")
            with open(self.path + ".tokens.json") as f:
                self._index = {t: i for i, t in enumerate(json.load(f))}
            if self._scores.shape[1:] != (len(LABEL_KEYS), N_ANCHORS):
                raise ValueError(f"{self.path}.npy: shape {self._scores.shape}")

    def __contains__(self, token: str) -> bool:
        self._open()
        return token in self._index

    def get(self, token: str) -> Optional[np.ndarray]:
        """[5, 256] float32 copy; None if the token is absent or its NC / DAC / TTC / C row is not finite."""
        self._open()
        i = self._index.get(token)
        if i is None:
            return None
        lab = np.array(self._scores[i], np.float32)
        if not np.isfinite(lab[list(PASS_IDX)]).all():
            return None
        if not self.ep_valid:
            lab[EP] = np.nan
        return lab

    def get_rows(self, tokens: Sequence[str]) -> np.ndarray:
        """[n, 5, 256] float32 for many tokens (sorted mmap read); absent tokens -> NaN rows."""
        self._open()
        rows = np.array([self._index.get(t, -1) for t in tokens], np.int64)
        out = np.full((len(rows), len(LABEL_KEYS), N_ANCHORS), np.nan, np.float32)
        have = np.flatnonzero(rows >= 0)
        if len(have):
            r = rows[have]
            srt = np.argsort(r, kind="stable")
            out[have[srt]] = np.asarray(self._scores[r[srt]], np.float32)
        if not self.ep_valid:
            out[:, EP] = np.nan
        return out

    def describe(self) -> Dict:
        p = Path(self.path + ".npy")
        st = p.stat()
        return {"kind": self.kind, "path": str(p), "ep_valid": self.ep_valid, "bytes": st.st_size,
                "mtime": int(st.st_mtime)}


class PackedGT:
    """GT (human) future of the packed split: CK_DATA/packed/<split>/{tokens.parquet, gt_traj.npy} (row aligned).
    gt_traj = scene.get_future_trajectory(8).poses (= v2 targets['trajectory']); NaN rows = unavailable."""

    def __init__(self, split: str, root: Union[str, Path] = C.CK_DATA):
        import pandas as pd
        d = Path(root) / "packed" / split
        self.split, self.dir = split, d
        self.tokens = [str(t) for t in pd.read_parquet(d / "tokens.parquet", columns=["token"]).token]
        self.gt = np.load(d / "gt_traj.npy", mmap_mode="r")
        if self.gt.shape != (len(self.tokens), C.N_POSE, 3):
            raise ValueError(f"{d}/gt_traj.npy: shape {self.gt.shape} vs {len(self.tokens)} tokens")
        self.row = {t: i for i, t in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    def get(self, token: str) -> Optional[np.ndarray]:
        i = self.row.get(token)
        if i is None:
            return None
        g = np.array(self.gt[i], np.float32)
        return g if np.isfinite(g).all() else None


# ----------------------------------------------------------------------------------------------- priors
def deploy_priors(split: str, rows: Optional[np.ndarray] = None, name: str = "cand",
                  root: Union[str, Path] = C.CK_DATA) -> Dict:
    """Mean official label of v2 r34's executed top-16 candidates (labels/<split>/<name>/labels.npy [N, 16, 9],
    ok.npy [N, 16]; rows = packed/<split>/tokens.parquet), over the given rows (default all), per PRIOR_KEYS."""
    d = Path(root) / "labels" / split / name
    lab = np.load(d / "labels.npy", mmap_mode="r")
    ok = np.load(d / "ok.npy")
    if rows is not None:
        rows = np.sort(np.asarray(rows, np.int64))
        lab, ok = np.asarray(lab[rows]), ok[rows]
    y = np.asarray(lab, np.float32)[ok][:, list(C.CK_LABEL_IDX)]           # [n, 5] CK_KEYS order
    out = {k: float(np.mean(y[:, j], dtype=np.float64)) for j, k in enumerate(LABEL_KEYS)}
    out["pass"] = float(np.mean((y[:, list(PASS_IDX)] >= 1.0 - 1e-6).all(1)))
    out["n_cands"] = int(ok.sum())
    return out


def split_priors(split: str, sampler: AnchorSampler, labels: Optional[LabelSource] = None,
                 gt: Optional[PackedGT] = None, deploy_name: str = "cand", root: Union[str, Path] = C.CK_DATA,
                 max_tokens: Optional[int] = None, chunk: int = 2048) -> Dict:
    """Deployment vs sampled vs population prior per key over a packed split, with log-odds offsets and label-shift
    weights.  'sampled' is the exact expectation over the sampling design (Sum pi y / Sum pi), not a Monte-Carlo
    estimate; it does not depend on epoch / seed."""
    labels = labels if labels is not None else (sampler.label_source or LabelSource())
    gt = gt if gt is not None else PackedGT(split, root)
    n = len(gt) if max_tokens is None else min(len(gt), int(max_tokens))
    nk = len(PRIOR_KEYS)
    s_num, p_num = np.zeros(nk), np.zeros(nk)
    g_num, g_den = np.zeros((3, nk)), np.zeros(3)
    s_den = p_den = 0.0
    fb = {k: 0 for k in FALLBACKS}
    n_pass_far = []
    used, skipped = [], {"no_labels": 0, "no_gt": 0}
    for c0 in range(0, n, chunk):
        toks = gt.tokens[c0:c0 + chunk]
        L = labels.get_rows(toks)
        G = np.asarray(gt.gt[c0:c0 + len(toks)], np.float32)
        for j, tok in enumerate(toks):
            lab = L[j]
            if not np.isfinite(lab[list(PASS_IDX)]).all():
                skipped["no_labels"] += 1
                continue
            if not np.isfinite(G[j]).all():
                skipped["no_gt"] += 1
                continue
            d = sampler.design(G[j], lab)
            y = np.concatenate([lab, d["pass"][None].astype(np.float32)], 0).astype(np.float64)    # [6, 256]
            pi = d["pi"]
            s_num += y @ pi
            s_den += pi.sum()
            p_num += y.sum(1)
            p_den += sampler.n_anchors
            grp = np.minimum(d["stratum"], GROUP_STRAT)          # far pass / fail -> strat
            for gi in range(3):
                m = grp == gi
                g_num[gi] += y[:, m] @ pi[m]
                g_den[gi] += pi[m].sum()
            fb[d["fallback"]] += 1
            n_pass_far.append(d["n_pass_far"])
            used.append(c0 + j)
    n_used = len(used)
    if n_used == 0:
        raise RuntimeError(f"{split}: no token with labels and GT")
    sampled = {k: float(s_num[i] / s_den) for i, k in enumerate(PRIOR_KEYS)}
    population = {k: float(p_num[i] / p_den) for i, k in enumerate(PRIOR_KEYS)}
    by_group = {GROUP_NAMES[gi]: {k: float(g_num[gi, i] / g_den[gi]) for i, k in enumerate(PRIOR_KEYS)}
                for gi in range(3)}
    deploy = deploy_priors(split, np.asarray(used), deploy_name, root)
    out = {"split": split, "n_tokens": n_used, "n_split": len(gt), "skipped": skipped, "sampler": sampler.config(),
           "label_source": labels.describe(), "deploy_source": str(Path(root) / "labels" / split / deploy_name),
           "deploy_n_cands": deploy.pop("n_cands"), "keys": list(PRIOR_KEYS),
           "deploy": deploy, "sampled": sampled, "population": population, "sampled_by_group": by_group,
           "offset_sampled": {k: log_odds_offset(deploy[k], sampled[k]) for k in PRIOR_KEYS},
           "offset_population": {k: log_odds_offset(deploy[k], population[k]) for k in PRIOR_KEYS},
           "offset_sampled_to_population": {k: log_odds_offset(population[k], sampled[k]) for k in PRIOR_KEYS},
           "lsw_sampled": {k: label_shift_weights(deploy[k], sampled[k]) for k in PRIOR_KEYS},
           "lsw_population": {k: label_shift_weights(deploy[k], population[k]) for k in PRIOR_KEYS},
           "fallback": {k: v / n_used for k, v in fb.items()},
           "n_pass_far_quantiles": {str(q): float(np.percentile(n_pass_far, q)) for q in (0, 5, 25, 50, 75, 95, 100)},
           "note": "EP prior = mean EP (continuous); its offset is indicative only. NaN for EP if the label source "
                   "is the WoTE fallback."}
    return out


def _main(argv=None):
    import argparse
    import os
    import time
    ap = argparse.ArgumentParser(description="CK2 raw-anchor sampling priors (deploy vs sampled) per split")
    ap.add_argument("--split", required=True, choices=["navtrain_train", "navtrain_val"])
    ap.add_argument("--out", default=str(Path(C.CK_DATA) / "ck2" / "priors"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--wote-labels", action="store_true", help="force the WoTE fallback label source (EP masked)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    src = LabelSource(prefer_official=not a.wote_labels)
    res = split_priors(a.split, AnchorSampler(label_source=src, seed=a.seed), labels=src, max_tokens=a.max_tokens)
    res["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(a.out, exist_ok=True)
    tag = "" if a.max_tokens is None else f"_n{a.max_tokens}"
    p = Path(a.out) / f"{a.split}_{src.kind}{tag}.json"
    tmp = p.with_name(f".{p.name}.tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, p)
    print(json.dumps({k: res[k] for k in ("n_tokens", "deploy", "sampled", "population", "offset_sampled",
                                          "offset_population", "fallback")}, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    _main()
