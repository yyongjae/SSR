"""Stage-T refiner data (IMPL_SPEC §3.8): teacher-cache access, per-split memmap packing, token-batched loading.

Sources (read-only; data root /home/external-user/ssd/yongjae_refiner = DATA_ROOT)
  run 4    : arm M reads resmap_cache.ResmapCache (ReSMap neck bev -> S grid), arm TM a ConcatTeacher of both
             ([512, 50, 100], det then map); arm_teachers(arm, split) builds them.  Token subsets (load_token_subset,
             restrict_rows) restrict packed rows to e.g. splits/train_trainlogs.parquet (AMENDMENT 6).
  teacher  : TeacherCache -- /home/external-user/datasets/teacher_cache/bevfusion/cache_{train,val}_50x100,
             samples/<tok[:2]>/<tok>.npz, ONLY the bev_feature member is read (np.load is lazy per member).
             Guard: the manifest checkpoint_sha256_head must equal TEACHER_SHA_HEAD = cddf943ffec8d6a8 and any path
             containing '_future' (cache_val_50x100_future = teacher runs on FUTURE frames) is refused.
             navtrain tokens (train + dev pools) -> cache_train_50x100; navtest -> cache_val_50x100.
  splits   : splits/<split>.parquet (token, log, frame_idx, map_location, part, split, fold, ...)
  human    : human/<split>.npz (extract_human.py; traj [8,3], v0, a0, eds [4], cmd, frame_gap, ...)
  drafts   : drafts/<split>/<token>.npz (make_draft_bank.py; drafts [13,8,3] f32, family [13] i8, params [13,6] f32,
             optional valid [13] bool -- all True if absent)
  labels   : scores/<split>.parquet (score_trajectories.py; one row per (token, k))
  objects  : objects/<split>/<token>.npz or objects/<subset>/<token>.npz (gt_future.save_objects)
  sdf      : sdf/<subset>/<token>.npz (sdf.save_sdf), subset = navtrain | navtest
  centerline: metric cache <root>/<log>/unknown/<token>/metric_cache.pkl (MC_ROOTS, first hit), route centerline
             (PDMPath, the one the official progress projects on).

Packed split (pack_split -> <DATA_ROOT>/packed/<split>/, PackedSplit reads it with np.load(mmap_mode='r'))
  index.parquet   row, token, log, fold, frame_idx      meta.json   parts, shapes, sources, timestamps
  done.npy        [N, len(PARTS)] u8 (1 = the part was written for that row; resumable per part and row)
  part 'human'     : human_traj [N,8,3] f32, v0 [N] f32, a0 [N] f32, eds [N,4] f32 (vx, vy, ax, ay), cmd [N] i8,
                     frame_gap [N] bool
  part 'drafts'    : drafts [N,13,8,3] f32, family [N,13] i8, params [N,13,6] f32, draft_valid [N,13] bool
  part 'labels'    : labels [N,13,len(LABEL_COLS)] f64 (LABEL_COLS order; float64 = the scorer's values exactly, so
                     eval_refiner can mix bank labels with new scores bitwise), pdm_progress_eff [N] f64
  part 'objects'   : obj_kf [N,A_MAX,11,6] f32, obj_first [N,A_MAX,6] f32, obj_meta [N,A_MAX,5] i16, obj_n [N] i32
                     (stored tracks, <= A_MAX), obj_dropped [N] i32 (tracks beyond A_MAX: the farthest from the GT ego
                     path, gt_future ordering), obj_n_kf [N] i16, obj_R [N] f32, obj_ego_kf [N,11,3] f32 (GT ego
                     rear-axle pose in N at the keyframes; UNKNOWN test / gt_ego of the surrogate)
  part 'sdf'       : sdf [N,320,256] f16 (E grid, sdf.py)
  part 'centerline': cl_xy [N,CL_MAX,2] f32, cl_valid [N,CL_MAX] bool, cl_n [N] i32 = the route-centerline VERTICES
                     in N of surrogate.centerline_from_metric_cache (the progress surrogate's own crop: arc-length
                     window [-CL_BACK, +CL_AHEAD] = [-30, +250] m around the projection of the t0 ego box centre,
                     +1 vertex each side; the
                     official progress projects the ego centre on the same PDMPath linestring), padded to CL_MAX = 1536
                     (integration 2026-09-28: the surrogate widened its crop from +150 to +250 m, which gives up to
                     1,153 vertices at 0.25 m (p50 1,078, 150 train tokens) -- the earlier CL_MAX = 1024 truncated it;
                     cl_n > CL_MAX would be truncated at the far end and is
                     recorded).  The N frame is the metric-cache rear-axle pose (== human/extract_human pose, 0.0 error).
Loader (TokenDataset + collate_tokens): one item = one token with all its K drafts; a batch = 8 tokens x 13 drafts.
  Batch dict (torch; T tokens, K drafts, A = max obj_n in the batch (>= 1)):
    tokens (list[str]), rows [T] i64, bev [T,256,50,100] f16 S grid ([T,512,50,100] for arm TM; None when the arm
    needs no BEV),
    tau0 [T,K,8,3] f32, draft_valid [T,K] bool, family [T,K] i64, params [T,K,6] f32,
    labels [T,K,L] f64, pdm_progress_eff [T] f64, human_traj [T,8,3], v0 [T], a0 [T], eds [T,4], cmd [T] i64,
    obj_kf [T,A,11,6] f32, obj_first [T,A,6] f32, obj_meta [T,A,5] i64, obj_valid [T,A] bool, obj_n_kf [T] i64,
    obj_R [T] f32, obj_ego_kf [T,11,3] f32, sdf [T,320,256] f16, cl_xy [T,L,2] f32, cl_valid [T,L] bool
    (L = max cl_n in the batch), cl_n [T] i64.
Folds / inner validation
  train split rows carry fold 0..4 (log-level, make_splits.py), dev fold = -1.  Cross-fitting run --fold k trains on
  folds != k (k = -1: all folds).  Early stopping uses an inner-validation subset of the TRAINING logs chosen by a
  hash of the log name (inner_val_logs: sha256(salt + log) < frac), identical for every arm and seed.

Deviations from IMPL_SPEC §3.8 (documented, interfaces unchanged)
  * Objects are padded to A_MAX = 800.  The full objects build (integration, 2026-09-28) found A up to 781 on the train
    split (14 of 24,000 tokens above the earlier 640; p99.9 601) and 592 on dev, so 800 drops nothing on train/dev;
    tracks beyond A_MAX (navtest later) would be the farthest from the GT ego path and are counted in obj_dropped.
    The loader pads a batch only to its own max obj_n, so the large pad costs disk, not compute.
  * The per-channel teacher z-score statistics are computed over (a seeded sample of) the tokens the run trains on
    (train split minus the held-out fold, minus the inner-validation logs), not over every train-split token, so an
    out-of-fold evaluation never sees statistics that include its own tokens.  n_norm = 2048 tokens by default.

CLI (resumable; parts can be run as their sources appear; rows with missing sources stay undone):
  python -m navsim.agents.para_ssr.refiner.data --split train [--parts human,centerline] [--workers 4]
Size: 418,762 B (~409 KiB) per token with A_MAX 800 / CL_MAX 1536 -> ~10.1 GB train (24k) + 3.4 GB dev (8k).  Loading: ~21 ms per token incl. the
teacher npz (2.6 MB) on the loaded machine, single process.
"""
from __future__ import annotations

import hashlib
import json
import lzma
import os
import pickle
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .adapters import BEV_H, BEV_W, IN_CH_T, compute_norm, teacher_to_s_grid

# ----------------------------------------------------------------------------------------------- constants
DATA_ROOT = Path("/home/external-user/ssd/yongjae_refiner")
REPO = Path("/home/external-user/yongjae/SSR")
TEACHER_ROOTS = {
    "navtrain": Path("/home/external-user/datasets/teacher_cache/bevfusion/cache_train_50x100"),
    "navtest": Path("/home/external-user/datasets/teacher_cache/bevfusion/cache_val_50x100"),
}
TEACHER_SHA_HEAD = "cddf943ffec8d6a8"
TEACHER_LAYOUT = "samples/<token[:2]>/<token>.npz"
SPLIT_SUBSET = {"train": "navtrain", "dev": "navtrain", "navtest": "navtest"}
MC_ROOTS = (
    DATA_ROOT / "metric_cache",
    REPO / "report/cause_and_correction_tests/E_train_split_feasibility/metric_cache",
    REPO / "data/exp/metric_cache",
)
K_DRAFTS = 13
A_MAX = 800              # >= 781 (train max) / 592 (dev max) tracks, full objects build 2026-09-28
N_KF = 11
E_H, E_W = 320, 256
CL_MAX = 1536            # padded centerline vertices (surrogate crop [-30, +250] m: <= 1,153 measured)
LABEL_COLS = ("nc", "dac", "ddc", "ep", "ttc", "comfort", "pdms", "raw_progress")
LBL = {c: i for i, c in enumerate(LABEL_COLS)}
PARTS = ("human", "drafts", "labels", "objects", "sdf", "centerline")
PART_ID = {p: i for i, p in enumerate(PARTS)}
TRAIN_PARTS = PARTS
PACK_VERSION = "refiner_pack_v2"          # v2: CL_MAX 1024 -> 1536 (surrogate crop +250 m), A_MAX 640 -> 800

# name -> (tail shape, dtype, part)
FIELDS: Dict[str, Tuple[Tuple[int, ...], str, str]] = {
    "human_traj": ((8, 3), "float32", "human"),
    "v0": ((), "float32", "human"),
    "a0": ((), "float32", "human"),
    "eds": ((4,), "float32", "human"),
    "cmd": ((), "int8", "human"),
    "frame_gap": ((), "bool", "human"),
    "drafts": ((K_DRAFTS, 8, 3), "float32", "drafts"),
    "family": ((K_DRAFTS,), "int8", "drafts"),
    "params": ((K_DRAFTS, 6), "float32", "drafts"),
    "draft_valid": ((K_DRAFTS,), "bool", "drafts"),
    "labels": ((K_DRAFTS, len(LABEL_COLS)), "float64", "labels"),      # float64: the scorer's exact values
    "pdm_progress_eff": ((), "float64", "labels"),
    "obj_kf": ((A_MAX, N_KF, 6), "float32", "objects"),
    "obj_first": ((A_MAX, 6), "float32", "objects"),
    "obj_meta": ((A_MAX, 5), "int16", "objects"),
    "obj_n": ((), "int32", "objects"),
    "obj_dropped": ((), "int32", "objects"),
    "obj_n_kf": ((), "int16", "objects"),
    "obj_R": ((), "float32", "objects"),
    "obj_ego_kf": ((N_KF, 3), "float32", "objects"),
    "sdf": ((E_H, E_W), "float16", "sdf"),
    "cl_xy": ((CL_MAX, 2), "float32", "centerline"),
    "cl_valid": ((CL_MAX,), "bool", "centerline"),
    "cl_n": ((), "int32", "centerline"),
}


# ----------------------------------------------------------------------------------------------- teacher cache
class TeacherCache:
    """Read-only access to one BEVFusion 50x100 teacher cache, pinned to the manifest checkpoint sha head."""

    def __init__(self, root: Union[str, Path], expect_sha_head: str = TEACHER_SHA_HEAD):
        root = Path(root)
        for p in (str(root), str(root.resolve())):
            if "_future" in p:
                raise ValueError(f"refusing teacher cache {p}: '*_future' caches hold FUTURE-frame teacher runs")
        mf = root / "manifest.json"
        if not mf.is_file():
            raise FileNotFoundError(f"teacher cache without manifest: {mf}")
        man = json.loads(mf.read_text())
        sha = str(man.get("checkpoint_sha256_head", ""))
        if sha != expect_sha_head:
            raise ValueError(f"teacher cache {root}: checkpoint_sha256_head {sha!r} != {expect_sha_head!r}")
        if man.get("layout") != TEACHER_LAYOUT:
            raise ValueError(f"teacher cache {root}: layout {man.get('layout')!r}")
        if list(man.get("target_bev_shape", [])) != [BEV_H, BEV_W] or int(man.get("bev_channels", -1)) != IN_CH_T:
            raise ValueError(f"teacher cache {root}: bev shape {man.get('target_bev_shape')} x {man.get('bev_channels')}")
        self.root = root
        self.manifest = man
        self.sha_head = sha

    @classmethod
    def for_subset(cls, subset: str) -> "TeacherCache":
        return cls(TEACHER_ROOTS[subset])

    def path(self, token: str) -> Path:
        return self.root / "samples" / token[:2] / f"{token}.npz"

    def has(self, token: str) -> bool:
        return self.path(token).is_file()

    def load_bev(self, token: str, s_grid: bool = True) -> np.ndarray:
        """bev_feature float16 [256, 50, 100]; S grid (col 0 = left) unless s_grid=False (cache layout)."""
        with np.load(self.path(token), allow_pickle=False) as z:
            x = z["bev_feature"]
        if x.shape != (IN_CH_T, BEV_H, BEV_W) or x.dtype != np.float16:
            raise ValueError(f"{token}: bev_feature {x.shape} {x.dtype}")
        return teacher_to_s_grid(x) if s_grid else x

    def load_det(self, token: str, s_grid: bool = True) -> Dict[str, np.ndarray]:
        """dense_heatmap [7, 50, 100] (S grid unless s_grid=False) + pred_boxes_3d [200, 9] (x_fwd, y_left, z, dx, dy,
        dz, yaw, vx, vy; N frame, lidar2ego = identity), pred_scores_3d [200], pred_labels_3d [200]."""
        with np.load(self.path(token), allow_pickle=False) as z:
            out = {k: z[k] for k in ("dense_heatmap", "pred_boxes_3d", "pred_scores_3d", "pred_labels_3d")}
        if s_grid:
            out["dense_heatmap"] = teacher_to_s_grid(out["dense_heatmap"])
        return out


def compute_teacher_norm(teacher: TeacherCache, tokens: Sequence[str], n_max: int = 2048, seed: int = 0):
    """Per-channel z-score statistics over a seeded sample of <= n_max tokens -> (mean [256], std [256], info)."""
    tokens = sorted(set(tokens))
    rng = np.random.default_rng(seed)
    pick = tokens if len(tokens) <= n_max else [tokens[i] for i in sorted(rng.choice(len(tokens), n_max, replace=False))]
    t = time.time()
    mean, std, info = compute_norm(teacher.load_bev(tk) for tk in pick)
    info.update(n_tokens=len(pick), n_pool=len(tokens), seed=seed, sec=round(time.time() - t, 1),
                teacher_root=str(teacher.root), sha_head=teacher.sha_head,
                token_hash=hashlib.sha256("\n".join(pick).encode()).hexdigest()[:16])
    return mean, std, info


# ----------------------------------------------------------------------------------------------- run-4 teachers
class ConcatTeacher:
    """Arm TM input (PRESTATED_DECISION_RULE AMENDMENT 6): load_bev(token) = concat([det.load_bev, map.load_bev]) ->
    float16 [512, 50, 100] on the S grid, det (BEVFusion TeacherCache) channels 0..255, map (ReSMap ResmapCache)
    256..511.  Picklable (both caches are).  det / map may be any object with load_bev (e.g. a shuffled view)."""

    def __init__(self, det, map_):
        self.det, self.map = det, map_
        self.root = det.root
        self.sha_head = det.sha_head
        self.map_sha_head = map_.sha_head

    def load_bev(self, token: str, s_grid: bool = True) -> np.ndarray:
        if not s_grid:
            raise ValueError("ConcatTeacher only serves the S grid (the two caches have different raw layouts)")
        d = np.asarray(self.det.load_bev(token, s_grid=True))
        m = np.asarray(self.map.load_bev(token, s_grid=True))
        if d.shape != (IN_CH_T, BEV_H, BEV_W) or m.shape != (IN_CH_T, BEV_H, BEV_W):
            raise ValueError(f"{token}: det {d.shape} / map {m.shape}")
        return np.concatenate([d.astype(np.float16, copy=False), m.astype(np.float16, copy=False)], 0)


def arm_teachers(arm: str, split: str, teacher_root=None, resmap_root=None):
    """-> (loader teacher, det cache, map cache) for an arm on a packed split (subset SPLIT_SUBSET[split]):
    none -> (None, None, None); T -> (det, det, None); M -> (map, None, map); TM -> (ConcatTeacher, det, map).
    det = BEVFusion TeacherCache (manifest sha-checked), map = ReSMap ResmapCache (meta sha256-checked; navtrain root =
    train_logs only, navtest root = the navtest tokens).  *_root overrides the cache directory (tests)."""
    from .resmap_cache import ResmapCache

    subset = SPLIT_SUBSET.get(split, "navtrain")
    det = mp = None
    if arm in ("T", "TM"):
        det = TeacherCache(teacher_root) if teacher_root else TeacherCache.for_subset(subset)
    if arm in ("M", "TM"):
        mp = ResmapCache(resmap_root) if resmap_root else ResmapCache.for_subset(subset)
    if arm == "none":
        return None, None, None
    if arm == "T":
        return det, det, None
    if arm == "M":
        return mp, None, mp
    if arm == "TM":
        return ConcatTeacher(det, mp), det, mp
    raise ValueError(f"arm {arm!r}")


def load_token_subset(path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
    """Token-subset parquet (column 'token'; e.g. splits/train_trainlogs.parquet) -> (unique tokens, info) with
    info = path, sha256 (of the file bytes), n_rows, n_tokens.  Deterministic; used by train_refiner --token-subset and
    eval_refiner (AMENDMENT 6)."""
    path = Path(path).resolve()
    b = path.read_bytes()
    df = pd.read_parquet(path, columns=["token"])
    tok = df.token.astype(str).to_numpy()
    info = dict(path=str(path), sha256=hashlib.sha256(b).hexdigest(), n_rows=int(len(df)), n_tokens=int(len(set(tok))))
    return np.unique(tok), info


def restrict_rows(packed: "PackedSplit", rows: np.ndarray, tokens) -> np.ndarray:
    """rows of the packed split whose token is in `tokens` (order kept)."""
    rows = np.asarray(rows, np.int64)
    return rows[np.isin(packed.index.token.values[rows].astype(str), np.asarray(list(tokens), dtype=str))]


# ----------------------------------------------------------------------------------------------- folds
def inner_val_logs(logs: Iterable[str], frac: float = 0.1, salt: str = "refiner_T_inner_val_v1") -> set:
    """Deterministic log-level inner-validation subset: sha256(salt + log) (first 60 bits, as a fraction) < frac.
    If that selects nothing (few logs) and frac > 0 with >= 2 logs, the log with the smallest hash is taken."""
    h = {lg: int(hashlib.sha256((salt + lg).encode()).hexdigest()[:15], 16) / float(16 ** 15) for lg in set(logs)}
    out = {lg for lg, v in h.items() if v < frac}
    if not out and frac > 0 and len(h) >= 2:
        out = {min(h, key=h.get)}
    return out


# ----------------------------------------------------------------------------------------------- metric cache
def locate_metric_cache(token: str, log: Optional[str], roots: Sequence[Path] = MC_ROOTS) -> Optional[Path]:
    if not log:
        return None
    for r in roots:
        p = Path(r) / log / "unknown" / token / "metric_cache.pkl"
        if p.is_file():
            return p
    return None


def load_metric_cache(path):
    with lzma.open(path, "rb") as f:
        return pickle.load(f)


def centerline_samples(mc) -> Dict[str, np.ndarray]:
    """Route-centerline vertices in the N frame, padded (part 'centerline'; surrogate.centerline_from_metric_cache)."""
    from .surrogate import centerline_from_metric_cache

    v = np.asarray(centerline_from_metric_cache(mc), np.float64)
    n = len(v)
    m = min(n, CL_MAX)
    xy = np.zeros((CL_MAX, 2), np.float32)
    ok = np.zeros(CL_MAX, bool)
    xy[:m], ok[:m] = v[:m], True
    return {"cl_xy": xy, "cl_valid": ok, "cl_n": np.int32(n)}


# ----------------------------------------------------------------------------------------------- sources
class Sources:
    """Where pack_split reads each part (defaults = the stage-T data root layout)."""

    def __init__(self, split: str, root: Union[str, Path] = DATA_ROOT, subset: Optional[str] = None,
                 human: Optional[Path] = None, drafts: Optional[Path] = None, scores: Optional[Path] = None,
                 objects: Optional[Sequence[Path]] = None, sdf: Optional[Path] = None,
                 mc_roots: Sequence[Path] = MC_ROOTS, centerline_fn: Optional[Callable] = None):
        root = Path(root)
        self.split = split
        self.subset = subset or SPLIT_SUBSET.get(split, "navtrain")
        self.human = Path(human) if human else root / "human" / f"{split}.npz"
        self.drafts = Path(drafts) if drafts else root / "drafts" / split
        self.scores = Path(scores) if scores else root / "scores" / f"{split}.parquet"
        self.objects = [Path(p) for p in objects] if objects else [root / "objects" / split, root / "objects" / self.subset]
        self.sdf = Path(sdf) if sdf else root / "sdf" / self.subset
        self.mc_roots = [Path(p) for p in mc_roots]
        self.centerline_fn = centerline_fn     # optional (token, log) -> dict | None (tests)

    def to_json(self) -> Dict:
        return dict(split=self.split, subset=self.subset, human=str(self.human), drafts=str(self.drafts),
                    scores=str(self.scores), objects=[str(p) for p in self.objects], sdf=str(self.sdf),
                    mc_roots=[str(p) for p in self.mc_roots])


def _read_drafts(src: Sources, token: str):
    p = src.drafts / f"{token}.npz"
    if not p.is_file():
        return None
    with np.load(p, allow_pickle=False) as z:
        d = np.asarray(z["drafts"], np.float32)
        fam = np.asarray(z["family"], np.int8) if "family" in z.files else np.full(len(d), -1, np.int8)
        par = np.asarray(z["params"], np.float32) if "params" in z.files else np.full((len(d), 6), np.nan, np.float32)
        val = np.asarray(z["valid"], bool) if "valid" in z.files else np.ones(len(d), bool)
    if d.shape != (K_DRAFTS, 8, 3):
        raise ValueError(f"{p}: drafts {d.shape} != ({K_DRAFTS}, 8, 3)")
    val = val & np.isfinite(d).all((1, 2))
    return {"drafts": d, "family": fam, "params": par, "draft_valid": val}


def _read_objects(src: Sources, token: str):
    from .gt_future import load_objects

    for d in src.objects:
        p = d / f"{token}.npz"
        if p.is_file():
            o = load_objects(p)
            n = int(o["kf"].shape[0])
            m = min(n, A_MAX)
            kf = np.zeros((A_MAX, N_KF, 6), np.float32)
            first = np.zeros((A_MAX, 6), np.float32)
            meta = np.zeros((A_MAX, 5), np.int16)
            kf[:m], first[:m], meta[:m] = o["kf"][:m], o["first"][:m], o["meta"][:m]
            return {"obj_kf": kf, "obj_first": first, "obj_meta": meta, "obj_n": np.int32(m),
                    "obj_dropped": np.int32(n - m), "obj_n_kf": np.int16(int(o.get("n_kf", N_KF))),
                    "obj_R": np.float32(float(o["R"])), "obj_ego_kf": np.asarray(o["ego_kf"], np.float32)}
    return None


def _read_sdf(src: Sources, token: str):
    from .sdf import load_sdf

    p = src.sdf / f"{token}.npz"
    return {"sdf": load_sdf(p)} if p.is_file() else None


def _read_centerline(src: Sources, token: str, log: str):
    if src.centerline_fn is not None:
        return src.centerline_fn(token, log)
    p = locate_metric_cache(token, log, src.mc_roots)
    if p is None:
        return None
    return centerline_samples(load_metric_cache(p))


_READERS = {"drafts": _read_drafts, "objects": _read_objects, "sdf": _read_sdf}


# ----------------------------------------------------------------------------------------------- packing
def _open_arrays(out: Path, names: Iterable[str], mode: str = "r+") -> Dict[str, np.ndarray]:
    return {n: np.load(out / f"{n}.npy", mmap_mode=mode) for n in names}


_PW: Dict[str, object] = {}


def _pack_worker_init(out: str, part: str, src: Sources):
    names = [n for n, f in FIELDS.items() if f[2] == part]
    _PW.update(arrays=_open_arrays(Path(out), names), part=part, src=src)


def _pack_chunk(task):
    """Worker: read one chunk of rows of one part and write them into the memmaps. -> (done rows, missing, errors)."""
    rows, tokens, logs = task
    A, part, src = _PW["arrays"], _PW["part"], _PW["src"]
    done, missing, errors = [], [], []
    for r, tk, lg in zip(rows, tokens, logs):
        try:
            d = _read_centerline(src, tk, lg) if part == "centerline" else _READERS[part](src, tk)
        except Exception as e:  # noqa: BLE001 -- recorded, row left undone
            errors.append((int(r), tk, f"{type(e).__name__}: {e}"[:300]))
            continue
        if d is None:
            missing.append(int(r))
            continue
        for n, v in d.items():
            A[n][r] = v
        done.append(int(r))
    for a in A.values():
        a.flush()
    return done, missing, errors


def _pack_human(src: Sources, idx: pd.DataFrame, arrays, done):
    if not src.human.is_file():
        return dict(missing=len(idx), note=f"no {src.human}")
    with np.load(src.human, allow_pickle=False) as z:
        pos = {t: i for i, t in enumerate(z["tokens"])}
        H = {k: z[k] for k in ("traj", "v0", "a0", "eds", "cmd", "frame_gap")}
    rows = np.array([r for r, t in zip(idx.row, idx.token) if t in pos], np.int64)
    src_i = np.array([pos[t] for t in idx.token if t in pos], np.int64)
    if len(rows):
        arrays["human_traj"][rows] = H["traj"][src_i]
        for k in ("v0", "a0", "eds", "cmd", "frame_gap"):
            arrays[k][rows] = H[k][src_i]
        done[rows, PART_ID["human"]] = 1
    return dict(done=int(len(rows)), missing=int(len(idx) - len(rows)))


def _pack_labels(src: Sources, idx: pd.DataFrame, arrays, done):
    if not src.scores.is_file():
        return dict(missing=len(idx), note=f"no {src.scores}")
    cols = ["token", "k", "pdm_progress_eff"] + [c for c in LABEL_COLS] + ["error"]
    df = pd.read_parquet(src.scores)
    df = df[[c for c in cols if c in df.columns]]
    pos = {t: r for r, t in zip(idx.row, idx.token)}
    df = df[df.token.isin(pos)]
    bad = set(df.loc[df["error"].fillna("").astype(str) != "", "token"]) if "error" in df.columns else set()
    df = df[(df.k >= 0) & (df.k < K_DRAFTS) & ~df.token.isin(bad)]
    cnt = df.groupby("token").k.nunique()
    full = [t for t, n in cnt.items() if n == K_DRAFTS]
    df = df[df.token.isin(full)].sort_values(["token", "k"])
    if len(full):
        rows = np.array([pos[t] for t in df.token.values[::K_DRAFTS]], np.int64)
        lab = df[list(LABEL_COLS)].to_numpy(np.float64).reshape(len(rows), K_DRAFTS, len(LABEL_COLS))
        arrays["labels"][rows] = lab
        arrays["pdm_progress_eff"][rows] = df.pdm_progress_eff.to_numpy(np.float64)[::K_DRAFTS]
        done[rows, PART_ID["labels"]] = 1
    return dict(done=len(full), missing=int(len(idx) - len(full)), token_errors=len(bad))


def pack_split(split: str, tokens: Optional[pd.DataFrame] = None, out_root: Union[str, Path] = DATA_ROOT / "packed",
               parts: Sequence[str] = PARTS, sources: Optional[Sources] = None, workers: int = 2, chunk: int = 64,
               limit: Optional[int] = None, redo: Sequence[str] = (), log_fn: Callable = print) -> Dict:
    """Pack (or resume packing) the given parts of one split into <out_root>/<split>/.  Returns a summary dict.

    tokens: DataFrame with token, log (+ fold, frame_idx); default splits/<split>.parquet.  Row order = that order.
    Rows whose source is missing stay undone (a rerun picks them up); `redo` parts are rewritten from scratch.
    """
    src = sources or Sources(split)
    out = Path(out_root) / split
    if tokens is None:
        tokens = pd.read_parquet(DATA_ROOT / "splits" / f"{split}.parquet")
    if limit is not None:
        tokens = tokens.iloc[:limit]
    idx = pd.DataFrame({"row": np.arange(len(tokens)), "token": tokens.token.astype(str).values,
                        "log": tokens.log.astype(str).values if "log" in tokens else "",
                        "fold": tokens.fold.astype(int).values if "fold" in tokens else -1,
                        "frame_idx": tokens.frame_idx.astype(int).values if "frame_idx" in tokens else -1})
    N = len(idx)
    out.mkdir(parents=True, exist_ok=True)
    idx_path = out / "index.parquet"
    if idx_path.exists():
        old = pd.read_parquet(idx_path)
        if list(old.token) != list(idx.token):
            raise ValueError(f"{out}: existing pack has a different token list/order ({len(old)} vs {N} rows)")
    else:
        idx.to_parquet(idx_path, index=False)
    for n, (tail, dt, _) in FIELDS.items():
        p = out / f"{n}.npy"
        if not p.exists():
            np.lib.format.open_memmap(p, mode="w+", dtype=np.dtype(dt), shape=(N,) + tail).flush()
        else:
            a = np.load(p, mmap_mode="r")
            if a.dtype != np.dtype(dt) or a.shape != (N,) + tail:
                raise ValueError(f"{p}: {a.dtype} {a.shape} != {dt} {(N,) + tail} (pack layout changed; delete it and "
                                 f"redo part {FIELDS[n][2]!r})")
    dpath = out / "done.npy"
    if not dpath.exists():
        np.save(dpath, np.zeros((N, len(PARTS)), np.uint8))
    done = np.load(dpath, mmap_mode="r+")
    for p in redo:
        done[:, PART_ID[p]] = 0
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else dict(version=PACK_VERSION, split=split, n=N,
                                                                                a_max=A_MAX, parts={})
    summary = {}
    for part in parts:
        t0 = time.time()
        todo = np.flatnonzero(done[:, PART_ID[part]] == 0)
        if part in ("human", "labels"):
            names = [n for n, f in FIELDS.items() if f[2] == part]
            arrays = _open_arrays(out, names)
            sub = idx.iloc[todo]
            res = (_pack_human if part == "human" else _pack_labels)(src, sub, arrays, done)
            for a in arrays.values():
                a.flush()
        else:
            tasks = [(todo[i:i + chunk], idx.token.values[todo[i:i + chunk]], idx.log.values[todo[i:i + chunk]])
                     for i in range(0, len(todo), chunk)]
            nd, nm, errs = 0, 0, []
            if tasks:
                if workers <= 1:
                    _pack_worker_init(str(out), part, src)
                    it = map(_pack_chunk, tasks)
                    pool = None
                else:
                    pool = get_context("fork").Pool(min(workers, 4), initializer=_pack_worker_init,
                                                    initargs=(str(out), part, src))
                    it = pool.imap_unordered(_pack_chunk, tasks)
                for i, (d, m, e) in enumerate(it):
                    if d:
                        done[np.asarray(d), PART_ID[part]] = 1
                    done.flush()
                    nd, nm, errs = nd + len(d), nm + len(m), errs + e
                    if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                        log_fn(f"[pack {split}/{part}] {i + 1}/{len(tasks)} chunks, done {nd}, missing {nm}, "
                               f"errors {len(errs)}, {time.time() - t0:.0f}s")
                if pool is not None:
                    pool.close()
                    pool.join()
            res = dict(done=nd, missing=nm, errors=len(errs), error_examples=errs[:5])
        done.flush()
        res.update(sec=round(time.time() - t0, 1), total_done=int(done[:, PART_ID[part]].sum()), n=N)
        summary[part] = res
        meta["parts"][part] = dict(res, updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
        log_fn(f"[pack {split}/{part}] {res}")
    meta["sources"] = src.to_json()
    meta["fields"] = {n: [list(f[0]), f[1], f[2]] for n, f in FIELDS.items()}
    meta["label_cols"] = list(LABEL_COLS)
    meta["centerline"] = dict(cl_max=CL_MAX, source="surrogate.centerline_from_metric_cache")
    tmp = meta_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=1, default=str))
    tmp.replace(meta_path)
    return summary


# ----------------------------------------------------------------------------------------------- reading
class PackedSplit:
    """Read-only view of <root>/<split>/ written by pack_split (memmaps)."""

    def __init__(self, split: str, root: Union[str, Path] = DATA_ROOT / "packed"):
        self.dir = Path(root) / split
        self.split = split
        self.index = pd.read_parquet(self.dir / "index.parquet")
        self.meta = json.loads((self.dir / "meta.json").read_text()) if (self.dir / "meta.json").exists() else {}
        self.done = np.load(self.dir / "done.npy", mmap_mode="r")
        self.arrays = _open_arrays(self.dir, FIELDS.keys(), mode="r")
        self.N = len(self.index)

    def rows_with(self, parts: Sequence[str] = TRAIN_PARTS) -> np.ndarray:
        ok = np.ones(self.N, bool)
        for p in parts:
            ok &= np.asarray(self.done[:, PART_ID[p]]) == 1
        if "human" in parts:
            ok &= ~np.asarray(self.arrays["frame_gap"], bool)
        return np.flatnonzero(ok)

    def select(self, parts: Sequence[str] = TRAIN_PARTS, folds: Optional[Sequence[int]] = None,
               exclude_folds: Optional[Sequence[int]] = None, logs_in: Optional[set] = None,
               logs_out: Optional[set] = None) -> np.ndarray:
        rows = self.rows_with(parts)
        f = self.index.fold.values[rows]
        lg = self.index.log.values[rows]
        keep = np.ones(len(rows), bool)
        if folds is not None:
            keep &= np.isin(f, list(folds))
        if exclude_folds is not None:
            keep &= ~np.isin(f, list(exclude_folds))
        if logs_in is not None:
            keep &= np.isin(lg, list(logs_in))
        if logs_out is not None:
            keep &= ~np.isin(lg, list(logs_out))
        return rows[keep]

    def row(self, r: int) -> Dict[str, np.ndarray]:
        return {n: np.array(a[r]) for n, a in self.arrays.items()}


class TokenDataset:
    """torch-style dataset: item i -> one token (all K drafts) of the packed split (+ teacher BEV if needed)."""

    def __init__(self, packed: PackedSplit, rows: Sequence[int], teacher: Optional[TeacherCache] = None):
        self.packed = packed
        self.rows = np.asarray(rows, np.int64)
        self.teacher = teacher

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict:
        r = int(self.rows[i])
        d = self.packed.row(r)
        d["row"] = r
        d["token"] = str(self.packed.index.token.values[r])
        if self.teacher is not None:
            d["bev"] = self.teacher.load_bev(d["token"], s_grid=True)
        return d


def collate_tokens(items: List[Dict]):
    """List of TokenDataset items -> batch dict of torch tensors (see module docstring)."""
    import torch

    st = lambda k, dt=None: torch.as_tensor(np.stack([it[k] for it in items]), dtype=dt)
    n_obj = max([int(it["obj_n"]) for it in items] + [1])
    n_cl = max([min(int(it["cl_n"]), CL_MAX) for it in items] + [2])
    b = {
        "tokens": [it["token"] for it in items],
        "rows": torch.as_tensor([it["row"] for it in items], dtype=torch.int64),
        "bev": st("bev") if "bev" in items[0] else None,
        "tau0": st("drafts"), "draft_valid": st("draft_valid"), "family": st("family", torch.int64),
        "params": st("params"), "labels": st("labels"), "pdm_progress_eff": st("pdm_progress_eff"),
        "human_traj": st("human_traj"), "v0": st("v0"), "a0": st("a0"), "eds": st("eds"), "cmd": st("cmd", torch.int64),
        "obj_kf": torch.as_tensor(np.stack([it["obj_kf"][:n_obj] for it in items])),
        "obj_first": torch.as_tensor(np.stack([it["obj_first"][:n_obj] for it in items])),
        "obj_meta": torch.as_tensor(np.stack([it["obj_meta"][:n_obj] for it in items]).astype(np.int64)),
        "obj_valid": torch.as_tensor(np.stack([np.arange(n_obj) < int(it["obj_n"]) for it in items])),
        "obj_n_kf": st("obj_n_kf", torch.int64), "obj_R": st("obj_R"), "obj_ego_kf": st("obj_ego_kf"),
        "sdf": st("sdf"), "cl_xy": torch.as_tensor(np.stack([it["cl_xy"][:n_cl] for it in items])),
        "cl_valid": torch.as_tensor(np.stack([it["cl_valid"][:n_cl] for it in items])), "cl_n": st("cl_n", torch.int64),
    }
    return b


def batch_to(batch: Dict, device) -> Dict:
    import torch

    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def make_loader(packed: PackedSplit, rows: Sequence[int], teacher: Optional[TeacherCache], tokens_per_batch: int = 8,
                shuffle: bool = True, seed: int = 0, workers: int = 1, drop_last: bool = True):
    """torch DataLoader over tokens (8 tokens x 13 drafts per batch).  Epoch order is seeded (set_epoch-free: the
    generator is re-seeded by the caller via loader.generator.manual_seed(seed + epoch))."""
    import torch
    from torch.utils.data import DataLoader

    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(TokenDataset(packed, rows, teacher), batch_size=tokens_per_batch, shuffle=shuffle,
                      collate_fn=collate_tokens, num_workers=min(int(workers), 2), drop_last=drop_last and shuffle,
                      generator=g, persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None)


# ----------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    """python -m navsim.agents.para_ssr.refiner.data --split train [--parts human,centerline] [--workers 4]"""
    import argparse

    ap = argparse.ArgumentParser(description="pack a stage-T split into memmaps (resumable)")
    ap.add_argument("--split", required=True, help="train | dev | navtest (tokens: <data>/splits/<split>.parquet)")
    ap.add_argument("--tokens", default=None, help="token parquet (token, log[, fold, frame_idx]); default by split")
    ap.add_argument("--parts", default=",".join(PARTS))
    ap.add_argument("--out-root", default=str(DATA_ROOT / "packed"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--redo", default="")
    a = ap.parse_args(argv)
    if a.workers > 4:
        raise SystemExit("--workers <= 4 (shared machine)")
    tok = pd.read_parquet(a.tokens) if a.tokens else None
    s = pack_split(a.split, tok, a.out_root, parts=[p for p in a.parts.split(",") if p], workers=a.workers,
                   chunk=a.chunk, limit=a.limit, redo=[p for p in a.redo.split(",") if p],
                   log_fn=lambda m: print(m, flush=True))
    print(json.dumps(s, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
