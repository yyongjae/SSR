"""CK2 e2e (v2 + CK2 student, SPEC ck2e2e §1-4 / §2 / §4) data side.  New file; the CK1 modules are imported unmodified.

Contents
  constants                     K16 = 16 (v2 top-k), NV = 6 (variants per parent, VNAMES order), G_K = 96 (= K16 * NV,
                                column c = k * 6 + v, k-major), streams, G_src codes, default paths
  type_balanced_cols / uniform_cols / draw_cols / sample_cols
                                per-token choice of n of the 80 non-identity variant columns (NU10; deterministic in
                                (valid, n, rng state); sample_cols keys the rng by (token, epoch, seed, stream))
  CK2E2ETargetBuilder(config)   target builder ('ck_e2e2_targets'): ck_row, GTLoader surrogate GT (ref_*), teacher
                                BEVs kd_bev_{0,1} / kd_ok_{0,1} and the warm-up arrays ck2_* (raw256 official labels
                                of all 256 anchors + all 96 columns of the fixed variant label files, as CK targets
                                with the EP of ck_e2e2.ep_target)
  WarmupSampler                 main-process warm-up draw (NU09): AnchorSampler K = 32 + n_var type-balanced variants
  rec2 chunks                   write_rec2_chunk / read_rec2_chunk (fmt 2: 96 trajectories per token); paths / DONE /
                                listing = e2e_data (format-agnostic layout io_dir/rec/epEEE/rR/c_<attempt>_<seq>.npz)
  CandRecorder2                 per-rank, per-epoch writer of the recorded 96 per token (CandRecorder logic)
  generation2                   open_generation2 / write_generation_rows2 (io_dir/lab/epEEE/{traj, labels, cand_ok,
                                valid, row_state}.npy + meta.json with format 'ck2_96', written last = ready)
  LabelStore2                   refresh(max_epoch) / lookup(rows, tokens, epoch, seed): 16 identities + 32 type-balanced
                                variants of the newest generation <= max_epoch (one-epoch lag, option B); rows without
                                one -> has False (the caller fills them with the warm-up fallback set, cands2.fill_fallback)
  ck2_warmup_prior              expected label mean of the student's warm-up mix (= ck2_dataset.ck2_label_prior, n_var 32)

EP target (ck_e2e2.ep_target, user 2026-10-08 ~20:10 KST; default EP_TARGET_DEFAULT = 'decoupled'): every conversion
of the 9 label columns (LABEL_COLS) into the 5 CK targets goes through ep_target.ck_targets -- warm-up ck2_raw_y /
ck2_var_y (wu_row), the on-policy generation labels G_y (LabelStore2.lookup) and the score prior (ck2_warmup_prior) --
so the label BCE target and the teacher -> student EP KD target (teachers trained with the same ep_target, checked by
tools/ck/e2e2/launch_util2.teacher_check) mean the same thing.  'decoupled' = official EP without the NC * DAC * DDC
factor (ck/ep_target.py).  The generation files keep the 9 official columns; the warm-up sampler's pass mask (NC / DAC /
TTC / C) never sees EP.

Every file / memmap / GTLoader is opened lazily per process (pid check); pickling drops them (DataLoader workers, fork or
spawn).  No collectives.  A structurally wrong data file (shape) raises at first open; a missing / unreadable ROW never
raises inside the builder: zeros + ck2_wu_ok False.  No torch-heavy imports (variants / decoder) here: the labeler and
the dataloader workers import this module.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from . import e2e_data as ED
from .anchor_sampler import GROUP_MID, GROUP_NEAR, GROUP_STRAT, AnchorSampler, load_anchors, token_rng
from .constants import CK_LABEL_IDX, LABEL_COLS
from .ep_target import EP_TARGETS, check_ep_target, ck_targets  # noqa: F401

PathLike = Union[str, Path]

# ----------------------------------------------------------------------------------------------- constants
K16, NV, G_K = 16, 6, 96                     # column c = k * NV + v (k = v2 top-16 rank, v = variant)
VNAMES = ("id", "a-1.0", "a-0.5", "a+0.5", "l-0.5", "l+0.5")     # == variants.variant_table(SPEEDS, LATS, 'separate')
N_TYPES = NV - 1                             # non-identity variant types 1..5
N_ROWS = ED.N_ROWS                           # 85109 packed navtrain_train rows
N_LAB = len(LABEL_COLS)                      # 9
CK_IDX = list(CK_LABEL_IDX)
N_WU_ANCHORS = 32                            # AnchorSampler K (8 near + 8 mid + 16 strat)
N_NEAR_MID = 16
GROUP_VAR = 3                                # warm-up group code of a variant candidate (anchors: 0 near, 1 mid, 2 strat)
STREAM_WU, STREAM_NOW, STREAM_LAB = "ck2e2e.wu", "ck2e2e.now", "ck2e2e.lab"
SRC_FALLBACK, SRC_PREV, SRC_OLDER, SRC_NODATA = 0, 1, 2, 3      # G_src codes
SRC_NAMES = ("fallback", "prev", "older", "nodata")
VAR_SAMPLING = ("type_balanced", "uniform")
REC2_FMT = 2
GEN2_FORMAT = "ck2_96"
GEN2_ORDER = ("c = k*6 + v; k = v2 top-16 rank (plan_final_rewards desc); "
              "v = id, a-1.0, a-0.5, a+0.5, l-0.5, l+0.5")

_CK = "/home/external-user/ssd/yongjae_refiner/ck"
WARMUP_DEFAULT: Dict[str, Any] = {
    "packed": f"{_CK}/packed/navtrain_train",
    "raw_labels": f"{_CK}/labels/navtrain_train/raw256",
    "var_dir": f"{_CK}/ck2/labels/navtrain_train/var_separate_sampler16_accstraight",
    "sampler_seed": 0, "n_var": 32, "sur_groups": "near_mid", "lat_kd_groups": "near_mid",
}
TEACHER_DET_RUN = f"{_CK}/ck2/train/ck2T10"            # user 2026-10-09 00:30 KST: official-EP teachers ck2T10 / ck2M10 + KD calibration (navtest: decoupled worse)
TEACHER_MAP_RUN = f"{_CK}/ck2/train/ck2M10"
EP_TARGET_DEFAULT = "official"                           # ck_e2e2.ep_target default (decoupled 10-08 20:10 -> official 10-09 00:30 KST)
REF_DATA_ROOT = ED.REF_DATA_ROOT
WU_TARGET_KEYS = ("ck2_wu_ok", "ck2_gt", "ck2_raw_y", "ck2_raw_ok", "ck2_var_traj", "ck2_var_y", "ck2_var_ok",
                  "ck2_var_valid", "ck2_var_anchor")


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def cfg2_dict(x) -> Dict[str, Any]:
    """ck_e2e2 as a plain dict from ParaSSRConfig (attribute ck_e2e2) / CKE2E2Config (to_dict) / DictConfig / dict /
    None ({} = defaults)."""
    if x is None:
        return {}
    if isinstance(x, dict):
        return dict(x)
    try:
        from omegaconf import DictConfig, OmegaConf
        if isinstance(x, DictConfig):
            return dict(OmegaConf.to_container(x, resolve=True))
    except ImportError:   # pragma: no cover
        pass
    if hasattr(x, "to_dict"):
        return dict(x.to_dict())
    if hasattr(x, "ck_e2e2"):
        return cfg2_dict(getattr(x, "ck_e2e2"))
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return dataclasses.asdict(x)
    return {}


def ep_target_of(x=None) -> str:
    """ck_e2e2.ep_target of anything cfg2_dict takes (missing, e.g. a bare warmup dict -> EP_TARGET_DEFAULT);
    validated."""
    v = cfg2_dict(x).get("ep_target")
    return check_ep_target(EP_TARGET_DEFAULT if v is None else v)


def warmup_cfg(x=None) -> Dict[str, Any]:
    """warm-up sub-config (WARMUP_DEFAULT merged with ck_e2e2.warmup); x = anything cfg2_dict takes, or the warmup
    dict itself (recognised by its 'var_dir' / 'raw_labels' / 'packed' keys)."""
    d = dict(WARMUP_DEFAULT)
    if isinstance(x, dict) and ({"var_dir", "raw_labels", "packed"} & set(x)) and "warmup" not in x:
        src = x
    else:
        src = cfg2_dict(x).get("warmup") or {}
    d.update({k: v for k, v in dict(src).items() if v is not None})
    d["sampler_seed"], d["n_var"] = int(d["sampler_seed"]), int(d["n_var"])
    for k in ("packed", "raw_labels", "var_dir"):
        d[k] = str(d[k])
    return d


# ----------------------------------------------------------------------------------------------- variant column draw
def _type_pools(valid: np.ndarray) -> List[np.ndarray]:
    v = np.asarray(valid, bool).reshape(-1, NV)
    return [np.flatnonzero(v[:, t]) * NV + t for t in range(1, NV)]          # ascending c per type 1..5


def type_balanced_cols(valid: np.ndarray, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """valid bool [16, 6] (or [96]; identity column ignored) -> (cols int64 [n] in 0..95, ok bool [n]).

    Quota n // 5 per type 1..5, the remaining n % 5 quotas +1 on the types rng.permutation(5)[:n % 5]; inside a type
    without replacement (rng.permutation); a type short of its quota -> the shortfall is filled uniformly without
    replacement from the remaining valid columns of all types; fewer than n valid columns in total -> padding by
    repetition (np.resize) with ok False; no valid column -> cols 0 (identity of rank 0), ok False.  Final order =
    rng.permutation(n).  Deterministic in (valid, n, rng state)."""
    n = int(n)
    if n < 0:
        raise ValueError(f"n {n} < 0")
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, bool)
    pools = _type_pools(valid)
    quota = np.full(N_TYPES, n // N_TYPES, np.int64)
    extra = n % N_TYPES
    if extra:
        quota[rng.permutation(N_TYPES)[:extra]] += 1
    taken, rest = [], []
    for t, pool in enumerate(pools):
        p = pool[rng.permutation(len(pool))]
        taken.append(p[:quota[t]])
        rest.append(p[quota[t]:])
    cols = np.concatenate(taken).astype(np.int64)
    short = n - len(cols)
    rest_all = np.concatenate(rest).astype(np.int64)
    if short > 0 and len(rest_all):
        cols = np.concatenate([cols, rest_all[rng.permutation(len(rest_all))[:short]]])
    nv = len(cols)
    if nv == 0:
        return np.zeros(n, np.int64), np.zeros(n, bool)
    ok = np.ones(n, bool)
    if nv < n:
        cols = np.resize(cols, n)
        ok = np.arange(n) < nv
    perm = rng.permutation(n)
    return cols[perm].astype(np.int64), ok[perm]


def uniform_cols(valid: np.ndarray, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """var_sampling 'uniform' = the CK2Dataset.variant_cols rule (ck2_dataset.py:190-204): valid non-identity columns
    in a random order, first n; fewer -> padding by repetition with ok False; none -> zeros, ok False."""
    n = int(n)
    v = np.asarray(valid, bool).reshape(-1, NV).copy()
    v[:, 0] = False
    cols = np.flatnonzero(v.reshape(-1))
    cols = cols[rng.permutation(len(cols))]
    nv = len(cols)
    if nv >= n:
        return cols[:n].astype(np.int64), np.ones(n, bool)
    if nv == 0:
        return np.zeros(n, np.int64), np.zeros(n, bool)
    return np.resize(cols, n).astype(np.int64), np.arange(n) < nv


def draw_cols(valid: np.ndarray, n: int, rng: np.random.Generator, mode: str = "type_balanced"):
    if mode == "type_balanced":
        return type_balanced_cols(valid, n, rng)
    if mode == "uniform":
        return uniform_cols(valid, n, rng)
    raise ValueError(f"var_sampling must be one of {VAR_SAMPLING}, got {mode!r}")


def sample_cols(valid96, tokens: Sequence[str], epoch: int, seed: int, stream: str, n: int,
                mode: str = "type_balanced") -> Tuple[np.ndarray, np.ndarray]:
    """valid96 bool [B, 96] (numpy or tensor), tokens [B] -> cols int64 [B, n], ok bool [B, n]; per token
    rng = token_rng(token, epoch, seed, stream) (independent of rank, worker and call order)."""
    v = _np(valid96).astype(bool).reshape(len(tokens), -1)
    cols = np.zeros((len(tokens), int(n)), np.int64)
    ok = np.zeros((len(tokens), int(n)), bool)
    for b, tok in enumerate(tokens):
        cols[b], ok[b] = draw_cols(v[b], n, token_rng(str(tok), int(epoch), int(seed), stream), mode)
    return cols, ok


# ----------------------------------------------------------------------------------------------- warm-up arrays
class WUSourceError(ValueError):
    """A warm-up source file has the wrong structure (shape / row count): a configuration error, always raised."""


class _WUArrays:
    """Lazy per-process memmaps of the warm-up sources (packed ok / gt, raw256 labels / ok, variant traj / labels /
    ok) + the index.npz members (valid, anchor_idx, built; npz members are loaded, ~11 MB).  ep_target: the EP of the
    CK targets wu_row derives from the 9 label columns (EP_TARGET_DEFAULT when not given)."""

    def __init__(self, wcfg: Dict[str, Any], ep_target: str = EP_TARGET_DEFAULT):
        self.cfg = dict(wcfg)
        self.ep_target = check_ep_target(ep_target)
        pk, rl, vd = Path(wcfg["packed"]), Path(wcfg["raw_labels"]), Path(wcfg["var_dir"])
        self.paths = {"p_ok": pk / "ok.npy", "gt": pk / "gt_traj.npy", "raw_y": rl / "labels.npy",
                      "raw_ok": rl / "ok.npy", "v_traj": vd / "traj.npy", "v_y": vd / "labels.npy",
                      "v_ok": vd / "ok.npy", "index": vd / "index.npz"}
        self._pid, self._mm = None, {}

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_pid"], d["_mm"] = None, {}
        return d

    SHAPES = {"p_ok": (), "gt": (8, 3), "raw_y": (256, N_LAB), "raw_ok": (256,), "v_traj": (G_K, 8, 3),
              "v_y": (G_K, N_LAB), "v_ok": (G_K,), "valid": (K16, NV), "anchor_idx": (K16,), "built": ()}

    def arr(self, name: str) -> np.ndarray:
        if self._pid != os.getpid():
            self._mm, self._pid = {}, os.getpid()
        a = self._mm.get(name)
        if a is None:
            if name in ("valid", "anchor_idx", "built"):
                with np.load(self.paths["index"], allow_pickle=False) as z:
                    for k in ("valid", "anchor_idx", "built"):
                        self._mm[k] = np.asarray(z[k])
                a = self._mm[name]
            else:
                a = self._mm[name] = np.load(self.paths[name], mmap_mode="r")
            want = self.SHAPES[name]
            if a.shape[1:] != want:
                raise WUSourceError(f"warm-up source {name} ({self.paths.get(name, self.paths['index'])}): shape "
                                 f"{a.shape}, expected [N, {', '.join(map(str, want))}]")
        return a

    def n_rows(self) -> int:
        return int(self.arr("p_ok").shape[0])

    def check(self, n_rows: Optional[int] = None) -> Dict[str, Any]:
        """Open everything (raises on a structural error), row counts equal, build.json variant table, and the
        provenance guard (provenance(): token order / anchors / sampler of the variant build vs the packed rows)."""
        n = self.n_rows() if n_rows is None else int(n_rows)
        for k in self.SHAPES:
            if self.arr(k).shape[0] != n:
                raise WUSourceError(f"warm-up source {k}: {self.arr(k).shape[0]} rows != {n}")
        b = json.loads((Path(self.cfg["var_dir"]) / "build.json").read_text())
        if list(b.get("names", [])) != list(VNAMES) or int(b.get("n_anchors", -1)) != K16:
            raise WUSourceError(f"{self.cfg['var_dir']}/build.json: names {b.get('names')} / n_anchors "
                             f"{b.get('n_anchors')} != {list(VNAMES)} / {K16}")
        return {"n_rows": n, "var_build": {k: b.get(k) for k in ("speeds", "lats", "combine", "s_on_frac", "names",
                                                                   "cfg_sha16", "traj_sha16", "merged")},
                "provenance": self.provenance(b)}

    def provenance(self, b: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Cheap row-identity guard (report 48 S56-5; the shape checks above cannot see a re-pack that keeps N but
        changes the token order): sha16 of the packed token list (U16, = label_variants / pack_v2 / label_cands) must
        equal the variant build.json token_sha16, packed/meta.json token_sha16 and raw_labels/shards/_plan.json
        token_sha16 (each when recorded); build.json anchors_sha16 == anchor_sampler.ANCHORS_SHA16 (load_anchors
        enforces the same on the file); build.json sampler == AnchorSampler(seed = build seed).config() minus
        label_source (the anchor subset of every variant row came from that sampler).  WUSourceError on a mismatch;
        -> what was checked.  ~0.1 s (85k tokens + the anchors file)."""
        import hashlib

        import pandas as pd
        from .anchor_sampler import ANCHORS_SHA16
        vd, pk, rl = Path(self.cfg["var_dir"]), Path(self.cfg["packed"]), Path(self.cfg["raw_labels"])
        if b is None:
            b = json.loads((vd / "build.json").read_text())
        toks = pd.read_parquet(pk / "tokens.parquet", columns=["token"]).token.astype(str).tolist()
        sha = hashlib.sha256(np.ascontiguousarray(np.array(toks, "U16")).tobytes()).hexdigest()[:16]
        out: Dict[str, Any] = {"token_sha16": sha, "n_tokens": len(toks)}
        refs = {"variant build.json": (vd / "build.json", b.get("token_sha16")),
                "packed meta.json": (pk / "meta.json", None), "raw_labels shards/_plan.json":
                (rl / "shards" / "_plan.json", None)}
        for what, (f, want) in refs.items():
            if want is None and f.is_file():
                want = (json.loads(f.read_text()) or {}).get("token_sha16")
            out[what] = want if want is not None else "not recorded"
            if want is not None and want != sha:
                raise WUSourceError(f"{f}: token_sha16 {want} != {sha} of {pk}/tokens.parquet (rows would not be the "
                                    f"same tokens: the sources were built from a different token order)")
        n_b = b.get("n_tokens")
        if n_b is not None and int(n_b) != len(toks):
            raise WUSourceError(f"{vd}/build.json: n_tokens {n_b} != {len(toks)} packed tokens")
        if b.get("anchors_sha16") is not None and b["anchors_sha16"] != ANCHORS_SHA16:
            raise WUSourceError(f"{vd}/build.json: anchors_sha16 {b['anchors_sha16']} != {ANCHORS_SHA16} (the anchor "
                                f"file the student and the teachers use)")
        out["anchors_sha16"] = b.get("anchors_sha16", "not recorded")
        if isinstance(b.get("sampler"), dict) and b.get("seed") is not None:
            want = {k: v for k, v in b["sampler"].items() if k != "label_source"}
            got = {k: v for k, v in AnchorSampler(seed=int(b["seed"])).config().items() if k != "label_source"}
            if want != got:
                diff = {k: (want.get(k), got.get(k)) for k in sorted(set(want) | set(got)) if want.get(k) != got.get(k)}
                raise WUSourceError(f"{vd}/build.json: anchor sampler config differs from the current AnchorSampler "
                                    f"(seed {b['seed']}): {diff}")
            out["sampler"] = "equal"
        else:
            out["sampler"] = "not recorded"
        return out

    def keep(self, r: int) -> bool:
        """CK2Dataset keep rule (ck2_dataset.py:100-124): packed ok & GT finite & raw256 ok on all 256 & variant row
        built & any variant ok."""
        A = self.arr
        return bool(A("p_ok")[r]) and bool(np.isfinite(np.asarray(A("gt")[r])).all()) and \
            bool(np.asarray(A("raw_ok")[r]).all()) and bool(A("built")[r]) and bool(np.asarray(A("v_ok")[r]).any())

    def keep_all(self) -> np.ndarray:
        """keep rule over all rows (vectorised; ~30 MB read)."""
        A = self.arr
        k = np.asarray(A("p_ok"), bool).copy()
        k &= np.isfinite(np.asarray(A("gt"))).all((1, 2))
        k &= np.asarray(A("raw_ok")).all(1)
        k &= np.asarray(A("built"), bool)
        k &= np.asarray(A("v_ok")).any(1)
        return k


def empty_wu_row() -> Dict[str, np.ndarray]:
    return {"ck2_wu_ok": np.bool_(False), "ck2_gt": np.zeros((8, 3), np.float32),
            "ck2_raw_y": np.zeros((256, 5), np.float32), "ck2_raw_ok": np.zeros(256, bool),
            "ck2_var_traj": np.zeros((G_K, 8, 3), np.float32), "ck2_var_y": np.zeros((G_K, 5), np.float32),
            "ck2_var_ok": np.zeros(G_K, bool), "ck2_var_valid": np.zeros(G_K, bool),
            "ck2_var_anchor": np.zeros(K16, np.int64)}


def _targets_wu(lab9: np.ndarray, ep_target: str) -> Tuple[np.ndarray, np.ndarray]:
    """[n, 9] labels -> (CK targets [n, 5] with non-finite rows zeroed, finite-target flag [n]).  A row whose 5 OFFICIAL
    CK columns are not all finite is zeroed whole (old rule) and flagged False even when its decoupled EP is finite;
    otherwise only a non-finite target element (decoupled EP of a non-finite raw_progress / pdm_progress_eff) is zeroed
    (flag False), so NC / DAC / TTC / C -- the warm-up sampler's pass mask -- are the official values for every
    ep_target.  'official' == the old lab9[:, CK_IDX] path bit for bit."""
    off_fin = np.isfinite(lab9[:, CK_IDX]).all(-1)
    y = ck_targets(lab9, ep_target)
    fin = np.isfinite(y).all(-1) & off_fin
    y = np.where(off_fin[:, None], np.where(np.isfinite(y), y, 0.0), 0.0).astype(np.float32)
    return y, fin


def wu_row(src: _WUArrays, r: int) -> Dict[str, np.ndarray]:
    """Warm-up arrays of one packed row (numpy); r < 0 or an unreadable row -> empty_wu_row() (wu_ok False).
    Non-finite values are zeroed; their ok flags are False.  ck2_raw_y / ck2_var_y = ck_targets(labels, src.ep_target)
    (_targets_wu)."""
    if r < 0:
        return empty_wu_row()
    A = src.arr
    try:
        if r >= src.n_rows():
            return empty_wu_row()
        gt = np.array(A("gt")[r], np.float32)
        ep = getattr(src, "ep_target", EP_TARGET_DEFAULT)
        raw, raw_fin = _targets_wu(np.array(A("raw_y")[r], np.float32), ep)
        vt = np.array(A("v_traj")[r], np.float32)
        vy, vy_fin = _targets_wu(np.array(A("v_y")[r], np.float32), ep)
        v_fin = vy_fin & np.isfinite(vt).all((-1, -2))
        out = {"ck2_wu_ok": np.bool_(src.keep(r)),
               "ck2_gt": np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0),
               "ck2_raw_y": raw,
               "ck2_raw_ok": np.asarray(A("raw_ok")[r], bool) & raw_fin,
               "ck2_var_traj": np.where(np.isfinite(vt), vt, 0.0).astype(np.float32),
               "ck2_var_y": np.where(v_fin[:, None], vy, 0.0).astype(np.float32),
               "ck2_var_ok": np.asarray(A("v_ok")[r], bool) & v_fin,
               "ck2_var_valid": np.asarray(A("valid")[r], bool).reshape(G_K),
               "ck2_var_anchor": np.asarray(A("anchor_idx")[r], np.int64)}
        if out["ck2_wu_ok"] and not np.isfinite(gt).all():          # pragma: no cover (keep() checks it)
            out["ck2_wu_ok"] = np.bool_(False)
        return out
    except WUSourceError:
        raise                                       # structural (shape) error: configuration, not a row problem
    except Exception:
        return empty_wu_row()


# ----------------------------------------------------------------------------------------------- target builder
try:
    from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder as _ATB
except Exception:   # pragma: no cover  (navsim always importable in the training env)
    _ATB = object


class CK2E2ETargetBuilder(_ATB):
    """Per-token CK2 e2e targets.  get_unique_name() = 'ck_e2e2_targets'.

    Outputs (B dim added by the collate):
      ck_row int64 [] (-1 = token not in packed navtrain_train)
      ref_* + ref_gt_ok (refiner.e2e.GTLoader surrogate GT)
      kd_bev_0 f16 [256,50,100] / kd_ok_0 bool [] (DET, BEVFusion t0); kd_bev_1 / kd_ok_1 (MAP, ReSMap) -- GTLoader
        rules, teacher-run order (det, map), arms from the runs' config.json
      ck2_wu_ok bool [] (CK2Dataset keep rule), ck2_gt f32 [8,3] (sampler GT), ck2_raw_y f32 [256,5] (raw256 official
        labels as CK targets, CK_KEYS, EP per ck_e2e2.ep_target), ck2_raw_ok bool [256], ck2_var_traj f32 [96,8,3],
        ck2_var_y f32 [96,5] (same rule), ck2_var_ok bool [96],
        ck2_var_valid bool [96] (index.npz valid, k-major), ck2_var_anchor int64 [16]
    Emitted for every token in every epoch (the workers do not know the epoch, NU09).
    """

    def __init__(self, config=None, ck_cfg=None, ref_data_root: Optional[str] = None):
        try:
            super().__init__()
        except TypeError:   # pragma: no cover
            pass
        c = cfg2_dict(ck_cfg if ck_cfg is not None else config)
        topk = int(c.get("topk", K16) or K16)
        if topk != K16:
            raise ValueError(f"CK2E2ETargetBuilder: topk {topk} != {K16} (generation layout is 16 x 6)")
        det = c.get("teacher_det_run", TEACHER_DET_RUN)
        mp = c.get("teacher_map_run", TEACHER_MAP_RUN)
        if not det:
            raise ValueError("CK2E2ETargetBuilder: teacher_det_run is required")
        self.teacher_runs: Tuple[str, ...] = tuple(str(r) for r in (det, mp) if r)
        root = ref_data_root or c.get("ref_data_root") or getattr(config, "ref_data_root", None) or REF_DATA_ROOT
        self.ref_data_root = str(root)
        self.wcfg = warmup_cfg(c)
        self.ep_target = ep_target_of(c)
        self.src = _WUArrays(self.wcfg, self.ep_target)
        self.rows = ED.RowMap(self.wcfg["packed"])
        self._gtl = None
        self._pid = None
        self.n_calls = 0

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_gtl"], d["_pid"] = None, None
        return d

    def get_unique_name(self) -> str:
        return "ck_e2e2_targets"

    def check(self) -> Dict[str, Any]:
        """Open every source once (raises on a structural error); main-process sanity check."""
        return self.src.check(len(self.rows))

    def _loader(self):
        if self._gtl is None or self._pid != os.getpid():
            from navsim.agents.para_ssr.refiner.e2e import GTLoader
            self._gtl = GTLoader(self.ref_data_root, self.teacher_runs)
            self._pid = os.getpid()
        return self._gtl

    def compute_targets(self, scene) -> Dict[str, torch.Tensor]:
        return self.compute_for_token(str(scene.scene_metadata.initial_token))

    def warmup_row(self, r: int) -> Dict[str, torch.Tensor]:
        return {k: torch.from_numpy(np.array(v, copy=True, order="C")) for k, v in wu_row(self.src, int(r)).items()}

    def compute_for_token(self, token: str) -> Dict[str, torch.Tensor]:
        self.n_calls += 1
        try:
            r = self.rows.row_of(token)
        except Exception:
            r = -1
        out: Dict[str, torch.Tensor] = {"ck_row": torch.tensor(int(r), dtype=torch.int64)}
        out.update(self._loader().load(str(token)))       # ref_* + kd_bev_i / kd_ok_i (GTLoader rules)
        out.update(self.warmup_row(r))
        return out


# ----------------------------------------------------------------------------------------------- warm-up sampler
class WarmupSampler:
    """Main-process warm-up draw (NU09), deterministic in (sampler_seed, seed, epoch, token):
      anchors   AnchorSampler(seed=sampler_seed).sample(token, ck2_gt, ck2_raw_y.T, epoch, shuffle=True) -> 32 ids
                (8 near + 8 mid + 16 strat, random slot order) = CK2Dataset.sample_anchors
      variants  draw_cols(ck2_var_valid [16, 6], n_var, token_rng(token, epoch, seed, stream), var_sampling)
    Rows with ck2_wu_ok False (or a sampler error) -> ok False, anc_idx 0, group = the canonical pattern
    [0] * 8 + [1] * 8 + [2] * 16 (so 'exactly 16 near+mid per token' always holds), var_cols 0, var_pad_ok False."""

    def __init__(self, anchors: Optional[np.ndarray] = None, sampler_seed: int = 0, seed: int = 0, n_var: int = 32,
                 var_sampling: str = "type_balanced"):
        if var_sampling not in VAR_SAMPLING:
            raise ValueError(f"var_sampling must be one of {VAR_SAMPLING}, got {var_sampling!r}")
        self.sampler = AnchorSampler(anchors=load_anchors() if anchors is None else anchors, seed=int(sampler_seed))
        if self.sampler.K != N_WU_ANCHORS or self.sampler.n_near + self.sampler.n_mid != N_NEAR_MID:
            raise ValueError(f"AnchorSampler K {self.sampler.K} / near+mid {self.sampler.n_near + self.sampler.n_mid}"
                             f" != {N_WU_ANCHORS} / {N_NEAR_MID}")
        self.seed, self.n_var, self.var_sampling = int(seed), int(n_var), var_sampling
        self.anchors = self.sampler.anchors                        # f32 [256, 8, 3]
        self._anc_t: Dict[str, torch.Tensor] = {}
        self._canon_group = np.concatenate([np.full(8, GROUP_NEAR), np.full(8, GROUP_MID),
                                            np.full(16, GROUP_STRAT)]).astype(np.int8)

    @classmethod
    def from_cfg(cls, cfg, anchors: Optional[np.ndarray] = None) -> "WarmupSampler":
        c = cfg2_dict(cfg)
        w = warmup_cfg(c)
        return cls(anchors, w["sampler_seed"], int(c.get("seed", 0)), w["n_var"],
                   str(c.get("var_sampling", "type_balanced")))

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_anc_t"] = {}
        return d

    def anchors_t(self, device) -> torch.Tensor:
        k = str(torch.device(device))
        if k not in self._anc_t:
            self._anc_t[k] = torch.from_numpy(self.anchors).to(device)
        return self._anc_t[k]

    def sample(self, tokens: Sequence[str], arrays: Dict[str, Any], epoch: int, stream: str = STREAM_WU,
               n_var: Optional[int] = None) -> Dict[str, np.ndarray]:
        """tokens [B]; arrays: ck2_wu_ok [B], ck2_gt [B,8,3], ck2_raw_y [B,256,5], ck2_var_valid [B,96] (numpy or
        tensors) -> anc_idx int64 [B,32], group int8 [B,32], var_cols int64 [B,n], var_pad_ok bool [B,n], ok bool [B],
        n_valid int64 [B] (valid non-identity variant columns)."""
        n = self.n_var if n_var is None else int(n_var)
        B = len(tokens)
        wu = _np(arrays["ck2_wu_ok"]).astype(bool).reshape(B)
        gt = _np(arrays["ck2_gt"]).astype(np.float32).reshape(B, 8, 3)
        ry = _np(arrays["ck2_raw_y"]).astype(np.float32).reshape(B, 256, 5)
        vv = _np(arrays["ck2_var_valid"]).astype(bool).reshape(B, G_K)
        out = {"anc_idx": np.zeros((B, N_WU_ANCHORS), np.int64),
               "group": np.tile(self._canon_group, (B, 1)),
               "var_cols": np.zeros((B, n), np.int64), "var_pad_ok": np.zeros((B, n), bool),
               "ok": np.zeros(B, bool), "n_valid": np.zeros(B, np.int64)}
        for b, tok in enumerate(tokens):
            if not wu[b]:
                continue
            try:
                s = self.sampler.sample(str(tok), gt[b], ry[b].T, epoch=int(epoch), shuffle=True)
            except (ValueError, AssertionError, KeyError):
                continue
            out["anc_idx"][b], out["group"][b] = s["idx"], s["group"]
            v = vv[b].reshape(K16, NV)
            out["n_valid"][b] = int(v[:, 1:].sum())
            out["var_cols"][b], out["var_pad_ok"][b] = draw_cols(v, n, token_rng(str(tok), int(epoch), self.seed,
                                                                                  stream), self.var_sampling)
            out["ok"][b] = True
        return out


# ----------------------------------------------------------------------------------------------- rec2 chunks
REC2_KEYS = ("row", "traj", "valid", "cand_idx", "v2_final", "v0", "gstep")


def _check_rec2(a: Dict[str, np.ndarray]) -> None:
    if "fmt" not in a or int(a["fmt"]) != REC2_FMT:
        raise ValueError(f"rec2 chunk: fmt {a.get('fmt')!r} != {REC2_FMT} (a CK1 chunk in a CK2 io_dir?)")
    n = a["row"].shape[0] if "row" in a else -1
    want = {"row": ((n,), np.int64), "traj": ((n, G_K, 8, 3), np.float32), "valid": ((n, G_K), np.bool_),
            "cand_idx": ((n, K16), np.int16), "v2_final": ((n, K16), np.float32), "v0": ((n,), np.float32),
            "gstep": ((n,), np.int64), "epoch": ((), np.int32), "rank": ((), np.int32), "fmt": ((), np.int32)}
    for name, (shape, dt) in want.items():
        if name not in a:
            raise ValueError(f"rec2 chunk: missing {name}")
        x = a[name]
        if tuple(x.shape) != shape or x.dtype != dt:
            raise ValueError(f"rec2 chunk {name}: {x.shape} {x.dtype} != {shape} {np.dtype(dt)}")


def _rec2_arrays(row, traj, valid, cand_idx, v2_final, v0, gstep, epoch, rank) -> Dict[str, np.ndarray]:
    row = _np(row).astype(np.int64).reshape(-1)
    n = len(row)
    a = {"row": row, "traj": _np(traj).astype(np.float32), "valid": _np(valid).astype(bool),
         "cand_idx": _np(cand_idx).astype(np.int16), "v2_final": _np(v2_final).astype(np.float32),
         "v0": _np(v0).astype(np.float32).reshape(-1),
         "gstep": np.broadcast_to(_np(gstep).astype(np.int64).reshape(-1), (n,)).copy(),
         "epoch": np.asarray(int(epoch), np.int32), "rank": np.asarray(int(rank), np.int32),
         "fmt": np.asarray(REC2_FMT, np.int32)}
    _check_rec2(a)
    return a


def write_rec2_chunk(path: PathLike, row, traj, valid, cand_idx, v2_final, v0, gstep, epoch: int, rank: int) -> Path:
    """Atomic (np.savez to '.{name}.tmp{pid}' in the same dir, then os.replace).  row int64 [n], traj f32 [n,96,8,3],
    valid bool [n,96], cand_idx int16 [n,16], v2_final f32 [n,16], v0 f32 [n], gstep int64 [n] (scalar broadcast),
    epoch / rank int32 [], fmt int32 [] = 2."""
    path = Path(path)
    a = _rec2_arrays(row, traj, valid, cand_idx, v2_final, v0, gstep, epoch, rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        np.savez(f, **a)
    os.replace(tmp, path)
    return path


def read_rec2_chunk(path: PathLike) -> Dict[str, Any]:
    """-> dict of the arrays (schema checked; fmt must be 2) + 'epoch_i' / 'rank_i' python ints."""
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    _check_rec2(a)
    a["epoch_i"], a["rank_i"] = int(a["epoch"]), int(a["rank"])
    return a


class CandRecorder2:
    """Per-rank, per-epoch writer of the recorded 96 trajectories per token (CandRecorder logic, online.py:424-492):
    chunks of chunk_tokens tokens -> write_rec2_chunk at e2e_data.rec_chunk_path(io_dir, epoch, rank, attempt, seq);
    close_epoch() flushes the rest and writes DONE_<attempt>.json (normal epoch end only).  attempt = 8 hex chars, new
    for every recorder (process x epoch start)."""

    def __init__(self, io_dir: PathLike, rank: int, world_size: int, epoch: int, chunk_tokens: int = 64,
                 attempt: Optional[str] = None):
        self.io_dir, self.rank, self.world_size, self.epoch = str(io_dir), int(rank), int(world_size), int(epoch)
        self.chunk_tokens = int(chunk_tokens)
        if self.chunk_tokens < 1:
            raise ValueError("chunk_tokens must be >= 1")
        self.attempt = attempt or uuid.uuid4().hex[:8]
        self.seq, self.n_tokens, self.closed = 0, 0, False
        self.chunks: List[str] = []
        self._buf: Dict[str, List[np.ndarray]] = {k: [] for k in REC2_KEYS}
        self._n_buf = 0

    def add(self, rows, traj96, valid96, cand_idx, v2_final, v0, gstep: int) -> int:
        """rows int64 [n] (row < 0 skipped), traj96 f32 [n,96,8,3] (exactly the trajectories of the step),
        valid96 bool [n,96], cand_idx [n,16], v2_final [n,16], v0 [n] -> number of tokens buffered."""
        if self.closed:
            raise RuntimeError("CandRecorder2.add after close_epoch")
        r = _np(rows).reshape(-1).astype(np.int64)
        keep = r >= 0
        n = int(keep.sum())
        if n == 0:
            return 0
        t = _np(traj96)
        if t.shape[1:] != (G_K, 8, 3):
            raise ValueError(f"CandRecorder2: traj96 {t.shape} != [n, {G_K}, 8, 3]")
        self._buf["row"].append(r[keep])
        self._buf["traj"].append(t[keep].astype(np.float32))
        self._buf["valid"].append(_np(valid96)[keep].astype(bool))
        self._buf["cand_idx"].append(_np(cand_idx)[keep].astype(np.int16))
        self._buf["v2_final"].append(_np(v2_final)[keep].astype(np.float32))
        self._buf["v0"].append(_np(v0).reshape(-1)[keep].astype(np.float32))
        self._buf["gstep"].append(np.full(n, int(gstep), np.int64))
        self._n_buf += n
        while self._n_buf >= self.chunk_tokens:
            self._write(self.chunk_tokens)
        return n

    def _write(self, n: int) -> None:
        cat = {k: np.concatenate(v, 0) for k, v in self._buf.items()}
        take = {k: v[:n] for k, v in cat.items()}
        rest = {k: v[n:] for k, v in cat.items()}
        path = Path(ED.rec_chunk_path(self.io_dir, self.epoch, self.rank, self.attempt, self.seq))
        write_rec2_chunk(path, take["row"], take["traj"], take["valid"], take["cand_idx"], take["v2_final"],
                         take["v0"], take["gstep"], self.epoch, self.rank)
        self.chunks.append(path.name)
        self.seq += 1
        self.n_tokens += n
        self._buf = {k: ([v] if len(v) else []) for k, v in rest.items()}
        self._n_buf -= n

    def flush(self) -> None:
        """write the partial chunk (no DONE)."""
        if self._n_buf > 0:
            self._write(self._n_buf)

    def close_epoch(self) -> None:
        if self.closed:
            return
        self.flush()
        ED.write_rec_done(self.io_dir, self.epoch, self.rank, self.world_size, self.attempt, list(self.chunks),
                          int(self.n_tokens))
        self.closed = True


# ----------------------------------------------------------------------------------------------- generations (96)
GEN2_FILES = {"traj": ((G_K, 8, 3), np.float32, np.nan), "labels": ((G_K, N_LAB), np.float32, np.nan),
              "cand_ok": ((G_K,), np.bool_, False), "valid": ((G_K,), np.bool_, False),
              "row_state": ((), np.uint8, 0)}
GEN2_KEYS = tuple(GEN2_FILES)


def _gen2_meta_ok(meta: Dict[str, Any], d: Path) -> None:
    if meta.get("format") != GEN2_FORMAT or int(meta.get("k", -1)) != G_K:
        raise ValueError(f"generation {d}: format {meta.get('format')!r} / k {meta.get('k')} is not "
                         f"{GEN2_FORMAT!r} / {G_K} (a CK1 generation in a CK2 io_dir?)")


def open_generation2(io_dir: PathLike, epoch: int, n_rows: int = N_ROWS, mode: str = "r") -> Dict[str, Any]:
    """Generation memmaps {traj f32 [N,96,8,3], labels f32 [N,96,9] (LABEL_COLS), cand_ok bool [N,96], valid bool
    [N,96], row_state uint8 [N], meta, dir} (as e2e_data.open_generation):
      'w+' labeler: creates the files under a FileLock (NaN / False / 0 filled, flushed; meta.json written LAST =
           ready marker, format 'ck2_96'); an existing generation is reopened 'r+' (idempotent restart).
      'r+' labeler restart (must exist).   'r' reader (must exist).
    A generation whose meta lacks format == 'ck2_96' is refused (ValueError)."""
    if mode not in ("w+", "r+", "r"):
        raise ValueError(f"open_generation2 mode {mode!r}")
    d = ED.gen_dir(io_dir, epoch)
    meta_p = d / "meta.json"
    if mode == "w+":
        with ED._FileLock(d / ".lock"):
            if not meta_p.is_file():
                d.mkdir(parents=True, exist_ok=True)
                for name, (shape, dt, fill) in GEN2_FILES.items():
                    mm = np.lib.format.open_memmap(d / f"{name}.npy", mode="w+", dtype=dt, shape=(int(n_rows),) + shape)
                    mm[...] = fill
                    mm.flush()
                    del mm
                ED._atomic_write_json(meta_p, {
                    "epoch": int(epoch), "n_rows": int(n_rows), "k": G_K, "format": GEN2_FORMAT, "order": GEN2_ORDER,
                    "vnames": list(VNAMES), "label_cols": list(LABEL_COLS),
                    "files": {n: [list(s), np.dtype(t).name] for n, (s, t, _) in GEN2_FILES.items()},
                    "row_state": "0 none, 1 done (written after traj/labels/cand_ok/valid)",
                    "created": time.time(), "utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())})
        mode = "r+"
    if not meta_p.is_file():
        raise FileNotFoundError(f"generation {d} not ready (no meta.json)")
    meta = json.loads(meta_p.read_text())
    _gen2_meta_ok(meta, d)
    if int(meta["n_rows"]) != int(n_rows):
        raise ValueError(f"{d}: n_rows {meta['n_rows']} != {n_rows}")
    out: Dict[str, Any] = {"meta": meta, "dir": d}
    for name, (shape, dt, _) in GEN2_FILES.items():
        a = np.load(d / f"{name}.npy", mmap_mode=mode)
        if a.shape != (int(n_rows),) + shape or a.dtype != dt:
            raise ValueError(f"{d}/{name}.npy: {a.shape} {a.dtype}")
        out[name] = a
    return out


def write_generation_rows2(gen: Dict[str, Any], rows, traj, labels, cand_ok, valid, overwrite: bool = False
                           ) -> np.ndarray:
    """Single-writer helper for the labeler: for rows not yet done (row_state != 1, unless overwrite) write traj
    [n,96,8,3], labels [n,96,9], cand_ok [n,96], valid [n,96]; flush; then row_state = 1; flush.  Duplicate rows in one
    call: the first occurrence wins.  -> bool [n] mask of the entries actually written."""
    rows = np.asarray(rows, np.int64).reshape(-1)
    traj, labels = np.asarray(traj, np.float32), np.asarray(labels, np.float32)
    cand_ok, valid = np.asarray(cand_ok, bool), np.asarray(valid, bool)
    n = len(rows)
    assert traj.shape == (n, G_K, 8, 3) and labels.shape == (n, G_K, N_LAB) and cand_ok.shape == (n, G_K) \
        and valid.shape == (n, G_K), (traj.shape, labels.shape, cand_ok.shape, valid.shape)
    st = gen["row_state"]
    _, first = np.unique(rows, return_index=True)
    w = np.zeros(n, bool)
    w[first] = True
    if not overwrite:
        w &= np.asarray(st[rows]) != 1
    if not w.any():
        return w
    r = rows[w]
    order = np.argsort(r)
    r, sel = r[order], np.flatnonzero(w)[order]
    gen["traj"][r] = traj[sel]
    gen["labels"][r] = labels[sel]
    gen["cand_ok"][r] = cand_ok[sel]
    gen["valid"][r] = valid[sel]
    for name in ("traj", "labels", "cand_ok", "valid"):
        gen[name].flush()
    st[r] = 1
    st.flush()
    return w


# ----------------------------------------------------------------------------------------------- label store
class LabelStore2:
    """G (GT BCE set) per packed row for the on-policy epochs (SPEC §4-3; option B, one-epoch lag).

    refresh(max_epoch) -> stats: per row src_epoch = the largest generation E <= max_epoch with row_state == 1 (-1 =
      none).  Pure read; the epoch-e caller passes max_epoch = e - label_lag.  Stats carry the old keys (max_epoch,
      gens, n_rows, n_prev, n_older, n_phase1 (== n_fallback), frac_*, lag_counts, sec) so online.label_supply_warning
      works unchanged.
    lookup(rows [B], tokens [B] | None, epoch, seed, n_orig=16, n_var=32) -> dict (torch, CPU unless device):
      G_traj f32 [B,48,8,3], G_y f32 [B,48,5] (ck_targets(9 label cols, ep_target): CK_KEYS, EP per ep_target),
      G_ok bool [B,48], G_vtype int8 [B,48]
      (0 identity, 1..5 variant type), G_col int16 [B,48] (generation column c, -1 = none), G_src int8 [B]
      (1 = the max_epoch generation, 2 = older, 0 = none -> fallback), G_epoch int16 [B] (-1 none), has bool [B].
      Generation rows: identity columns k * 6 (k = 0..15, cand_ok & finite) + type_balanced_cols(valid & cand_ok, 32,
      token_rng(token, epoch, seed, 'ck2e2e.lab')) (NU11).  Rows without a generation (or row < 0, or out of range):
      has False, zeros, ok False -> the caller fills them (cands2.fill_fallback, NU12).  Non-finite values are zeroed
      (ok False).  tokens None -> taken from the packed RowMap.
    Lazy per-process memmaps; pickling drops them and the src index (re-read on first use after a pid change).
    """

    def __init__(self, io_dir: PathLike, n_rows: int = N_ROWS, packed: PathLike = WARMUP_DEFAULT["packed"],
                 var_sampling: str = "type_balanced", ep_target: str = EP_TARGET_DEFAULT):
        if var_sampling not in VAR_SAMPLING:
            raise ValueError(f"var_sampling must be one of {VAR_SAMPLING}")
        self.ep_target = check_ep_target(ep_target)
        self.io_dir = Path(io_dir)
        self.n_rows = int(n_rows)
        self.packed = str(packed)
        self.var_sampling = var_sampling
        self.rowmap = ED.RowMap(self.packed)
        self.max_epoch: Optional[int] = None
        self.src_epoch: Optional[np.ndarray] = None
        self.last_stats: Dict[str, Any] = {}
        self._gens: Dict[int, Dict[str, Any]] = {}
        self._pid = None
        self.cum = self._zero_cum()

    @staticmethod
    def _zero_cum() -> Dict[str, int]:
        return {"lookup_rows": 0, "src_fallback": 0, "src_prev": 0, "src_older": 0, "no_row": 0}

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_gens"], d["_pid"], d["src_epoch"] = {}, None, None
        return d

    def _proc(self):
        if self._pid != os.getpid():
            self._gens, self._pid = {}, os.getpid()
            if self.max_epoch is not None and self.src_epoch is None:
                self.refresh(self.max_epoch)

    def _gen(self, epoch: int) -> Dict[str, Any]:
        g = self._gens.get(epoch)
        if g is None:
            g = self._gens[epoch] = open_generation2(self.io_dir, epoch, self.n_rows, "r")
        return g

    def refresh(self, max_epoch: int) -> Dict[str, Any]:
        t0 = time.time()
        if self._pid != os.getpid():
            self._gens, self._pid = {}, os.getpid()
        max_epoch = int(max_epoch)
        src = np.full(self.n_rows, -1, np.int16)
        gens = [e for e in ED.list_generations(self.io_dir) if e <= max_epoch]
        used = []
        for e in sorted(gens, reverse=True):
            gd = ED.gen_dir(self.io_dir, e)
            try:
                meta = json.loads((gd / "meta.json").read_text())
            except (OSError, ValueError):
                continue
            _gen2_meta_ok(meta, gd)
            try:
                st = np.array(np.load(gd / "row_state.npy", mmap_mode="r"))
            except (OSError, ValueError):
                continue
            if st.shape != (self.n_rows,):
                raise ValueError(f"generation ep{e:03d}: row_state {st.shape} != ({self.n_rows},)")
            m = (src < 0) & (st == 1)
            src[m] = e
            used.append(e)
            self._gens.pop(e, None)
        self.max_epoch, self.src_epoch = max_epoch, src
        n_prev = int(((src == max_epoch) & (src >= 0)).sum()) if max_epoch >= 0 else 0
        n_any = int((src >= 0).sum())
        lag = {}
        for e in used:
            c = int((src == e).sum())
            if c:
                lag[str(max_epoch - e + 1)] = c
        nf = self.n_rows - n_any
        st = {"max_epoch": max_epoch, "gens": sorted(used), "n_rows": self.n_rows, "n_prev": n_prev,
              "n_older": n_any - n_prev, "n_fallback": nf, "n_phase1": nf,
              "frac_prev": n_prev / self.n_rows, "frac_older": (n_any - n_prev) / self.n_rows,
              "frac_fallback": nf / self.n_rows, "frac_phase1": nf / self.n_rows, "lag_counts": lag,
              "sec": round(time.time() - t0, 4)}
        self.last_stats = st
        return st

    def lookup(self, rows, tokens: Optional[Sequence[str]] = None, epoch: int = 0, seed: int = 0,
               n_orig: int = K16, n_var: int = 32, device=None) -> Dict[str, torch.Tensor]:
        self._proc()
        r = _np(rows).astype(np.int64).reshape(-1)
        B = len(r)
        if n_orig != K16:
            raise ValueError(f"LabelStore2.lookup: n_orig must be {K16} (all identities), got {n_orig}")
        if tokens is None:
            tokens = [self.rowmap.token_log(int(x))[0] if 0 <= int(x) < len(self.rowmap) else "" for x in r]
        if len(tokens) != B:
            raise ValueError(f"lookup: {len(tokens)} tokens for {B} rows")
        KG = n_orig + int(n_var)
        traj = np.zeros((B, KG, 8, 3), np.float32)
        y = np.zeros((B, KG, 5), np.float32)
        ok = np.zeros((B, KG), bool)
        vt = np.zeros((B, KG), np.int8)
        col = np.full((B, KG), -1, np.int16)
        src = np.full(B, SRC_FALLBACK, np.int8)
        gep = np.full(B, -1, np.int16)
        has = np.zeros(B, bool)
        se = self.src_epoch
        ident = np.arange(K16, dtype=np.int64) * NV
        for b, rr in enumerate(r):
            self.cum["lookup_rows"] += 1
            if rr < 0 or rr >= self.n_rows:
                self.cum["no_row"] += 1
                self.cum["src_fallback"] += 1
                continue
            e = int(se[rr]) if se is not None else -1
            if e < 0:
                self.cum["src_fallback"] += 1
                continue
            g = self._gen(e)
            t = np.asarray(g["traj"][rr], np.float32)                       # [96, 8, 3]
            lab = ck_targets(np.asarray(g["labels"][rr], np.float32), self.ep_target)     # [96, 5]
            cok = np.asarray(g["cand_ok"][rr], bool)
            val = np.asarray(g["valid"][rr], bool)
            cols, pad = draw_cols((val & cok).reshape(K16, NV), int(n_var),
                                  token_rng(str(tokens[b]), int(epoch), int(seed), STREAM_LAB), self.var_sampling)
            c = np.concatenate([ident, cols])
            o = np.concatenate([cok[ident], cok[cols] & val[cols] & pad])
            tt, yy = t[c], lab[c]
            fin_t = np.isfinite(tt).all((-1, -2))
            fin_y = np.isfinite(yy).all(-1)
            traj[b] = np.where(fin_t[:, None, None], tt, 0.0)
            y[b] = np.where(fin_y[:, None], yy, 0.0)
            ok[b] = o & fin_t & fin_y
            vt[b] = (c % NV).astype(np.int8)
            col[b] = c.astype(np.int16)
            src[b] = SRC_PREV if e == self.max_epoch else SRC_OLDER
            gep[b] = e
            has[b] = True
            self.cum["src_prev" if e == self.max_epoch else "src_older"] += 1
        out = {"G_traj": torch.from_numpy(traj), "G_y": torch.from_numpy(y), "G_ok": torch.from_numpy(ok),
               "G_vtype": torch.from_numpy(vt), "G_col": torch.from_numpy(col), "G_src": torch.from_numpy(src),
               "G_epoch": torch.from_numpy(gep), "has": torch.from_numpy(has)}
        if device is not None:
            out = {k: v.to(device, non_blocking=True) for k, v in out.items()}
        return out

    def stats(self, reset: bool = False) -> Dict[str, Any]:
        """Cumulative lookup source counts since the last reset (+ last refresh stats)."""
        c = dict(self.cum)
        n = max(c["lookup_rows"], 1)
        c.update({"frac_fallback": c["src_fallback"] / n, "frac_prev": c["src_prev"] / n,
                  "frac_older": c["src_older"] / n, "refresh": dict(self.last_stats)})
        if reset:
            self.cum = self._zero_cum()
        return c


# ----------------------------------------------------------------------------------------------- prior
def ck2_warmup_prior(cfg=None, n_max: int = 4000, n_var: Optional[int] = None,
                     ep_target: Optional[str] = None) -> Optional[np.ndarray]:
    """Expected unweighted label mean per CK key of the student's warm-up mix (NU20): over <= n_max evenly spaced
    keep-rows, epoch 0: Sum y over the 32 sampled anchors (y_ok) + min(1, n_var / n_valid) x Sum y over every valid
    non-identity variant column (var_ok), divided by the matching counts; clipped to [1e-3, 1 - 1e-3].  Same
    arithmetic (and result) as tools/ck/data/ck2_dataset.ck2_label_prior(CK2Dataset(n_var=n_var, var_name=<var_dir>,
    sampler_seed, ep_target)).  cfg: anything cfg2_dict / warmup_cfg takes (None -> defaults).  y = ck_targets(labels,
    ep_target) (None -> ck_e2e2.ep_target of cfg, default EP_TARGET_DEFAULT; the sampler sees the official labels).
    None if nothing usable."""
    w = warmup_cfg(cfg)
    ep = check_ep_target(ep_target) if ep_target is not None else ep_target_of(cfg)
    nv_draw = int(w["n_var"] if n_var is None else n_var)
    src = _WUArrays(w)
    A = src.arr
    try:
        rows = np.flatnonzero(src.keep_all())
    except FileNotFoundError:
        return None
    n = len(rows)
    if n == 0:
        return None
    rm = ED.RowMap(w["packed"])
    smp = AnchorSampler(anchors=load_anchors(), seed=int(w["sampler_seed"]))
    ii = np.unique(np.linspace(0, n - 1, min(int(n_max), n)).astype(np.int64))
    num, den = np.zeros(5), 0.0
    for i in ii:
        r = int(rows[i])
        tok = rm.token_log(r)[0]
        lab = np.array(A("raw_y")[r], np.float32)                                     # [256, 9]
        gt = np.asarray(A("gt")[r], np.float32)
        s = smp.sample(tok, gt, lab[:, CK_IDX].T, epoch=0, shuffle=True)
        idx = s["idx"]
        L = lab[idx]
        yy = np.ascontiguousarray(ck_targets(L, ep))
        m = np.asarray(A("raw_ok")[r, idx], bool) & np.isfinite(yy).all(1)
        num += yy[m].sum(0)
        den += m.sum()
        if nv_draw > 0:
            valid = np.asarray(A("valid")[r], bool).copy()
            valid[:, 0] = False
            cols = np.flatnonzero(valid.reshape(-1))
            if len(cols):
                vl = ck_targets(np.asarray(A("v_y")[r, cols], np.float64), ep)
                vo = np.asarray(A("v_ok")[r, cols], bool) & np.isfinite(vl).all(1)
                f = min(1.0, nv_draw / len(cols))
                num += f * vl[vo].sum(0)
                den += f * vo.sum()
    if den <= 0:
        return None
    return np.clip(num / den, 1e-3, 1 - 1e-3)


def tokens_of_rows(rows, packed: PathLike = WARMUP_DEFAULT["packed"]) -> List[str]:
    """packed rows -> tokens ('' for rows < 0 / out of range)."""
    rm = ED._rowmap(packed)
    n = len(rm)
    tk = rm.tokens
    return [str(tk[int(x)]) if 0 <= int(x) < n else "" for x in _np(rows).reshape(-1)]
