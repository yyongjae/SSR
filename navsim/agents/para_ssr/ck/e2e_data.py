"""CK Phase 2 (v2 + CK e2e, report 45; spec contract_e2e.json 'data') data side.

Owned by data.  Contents
  CKE2ETargetBuilder(config)        target builder (dataloader workers): ck_row, GTLoader surrogate GT (ref_*), teacher
                                    BEVs kd_bev_{0,1} / kd_ok_{0,1} (DET = BEVFusion t0, MAP = ReSMap) and the Phase 1
                                    replay arrays ck_p1_* (r34 top-16, official labels, offline KD targets).
  record chunks                     rec_chunk_path / write_rec_chunk / read_rec_chunk / write_rec_done (+ list helpers):
                                    {io_dir}/rec/ep{E:03d}/r{R}/c_<attempt>_<seq:06d>.npz, DONE_<attempt>.json
  generations                       gen_dir / open_generation / write_generation_rows / list_generations:
                                    {io_dir}/lab/ep{E:03d}/{traj,labels,cand_ok,row_state}.npy + meta.json (meta.json is
                                    written last at creation = "ready"; row_state is written after the row's data)
  LabelStore(io_dir, phase1_cfg)    refresh(max_epoch) / lookup(rows): per row the newest generation E <= max_epoch with
                                    row_state == 1, else the Phase 1 set (cand + kd_corr, official labels).
  RowMap / row_of / token_log       packed/navtrain_train/tokens.parquet row <-> token / log
  phase1_label_prior(n_max)         = tools/ck/train_ck.label_prior('navtrain_train')

Every file / memmap / GTLoader is opened lazily per process (pid check); pickling drops them (DataLoader workers,
fork or spawn).  No collectives anywhere.  Missing rows / unreadable files never raise inside the builder: zeros + the
matching ok flag False (GTLoader rules for the surrogate GT and the teacher BEVs).
"""
from __future__ import annotations

import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .constants import CK_LABEL_IDX, LABEL_COLS

PathLike = Union[str, Path]

# ----------------------------------------------------------------------------------------------- constants
N_ROWS = 85109                      # len(packed/navtrain_train/tokens.parquet)
K = 16                              # recorded candidates (= ck_e2e.topk)
G_K = 32                            # generation trajectories per row = cat(recorded cand 16, recorded tau'_KD 16)
N_LAB = len(LABEL_COLS)             # 9
CK_IDX = list(CK_LABEL_IDX)
BEV_SHAPE = (256, 50, 100)

PHASE1_DEFAULT = {
    "packed": "/home/external-user/ssd/yongjae_refiner/ck/packed/navtrain_train",
    "labels_cand": "/home/external-user/ssd/yongjae_refiner/ck/labels/navtrain_train/cand",
    "labels_kd_corr": "/home/external-user/ssd/yongjae_refiner/ck/labels/navtrain_train/kd_corr",
    "kd_targets": "/home/external-user/ssd/yongjae_refiner/ck/kd_targets/phase1",
}
TEACHER_DET_RUN = "/home/external-user/ssd/yongjae_refiner/ck/train/ckT_p1"
TEACHER_MAP_RUN = "/home/external-user/ssd/yongjae_refiner/ck/train/ckM_p1"
REF_DATA_ROOT = "/home/external-user/ssd/yongjae_refiner"

SRC_PHASE1, SRC_PREV, SRC_OLDER = 0, 1, 2      # G_src codes

_CHUNK_RE = re.compile(r"^c_([0-9a-f]{8})_(\d{6})\.npz$")
_EP_RE = re.compile(r"^ep(\d{3})$")
_RANK_RE = re.compile(r"^r(\d+)$")


# ----------------------------------------------------------------------------------------------- small utils
def _atomic_write_json(path: PathLike, obj, indent: int = 1) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


class _FileLock:
    """fcntl advisory lock (same semantics as tools/ck/ckutil.FileLock; no tools import from navsim)."""

    def __init__(self, path: PathLike):
        self.path = Path(path)

    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a+")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        import fcntl
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()
        return False


def _cfg_dict(x) -> Dict[str, Any]:
    """ck_e2e as a plain dict from ParaSSRConfig / CKE2EConfig (to_dict) / DictConfig / dict / None."""
    import dataclasses
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
    if hasattr(x, "to_dict"):                       # CKE2EConfig
        return dict(x.to_dict())
    if hasattr(x, "ck_e2e"):                        # ParaSSRConfig (or a namespace holding ck_e2e)
        return _cfg_dict(getattr(x, "ck_e2e"))
    if dataclasses.is_dataclass(x) and hasattr(x, "teacher_det_run"):
        return dataclasses.asdict(x)
    return {}                                       # a config without ck_e2e -> defaults


def phase1_paths(phase1_cfg=None) -> Dict[str, Path]:
    d = dict(PHASE1_DEFAULT)
    if phase1_cfg:
        d.update({k: v for k, v in dict(phase1_cfg).items() if v})
    return {k: Path(v) for k, v in d.items()}


# ----------------------------------------------------------------------------------------------- row map
class RowMap:
    """packed tokens.parquet (token, log, row) <-> row; lazy per process, picklable."""

    def __init__(self, packed: PathLike = PHASE1_DEFAULT["packed"]):
        self.packed = Path(packed)
        self._pid = None
        self._idx: Dict[str, int] = {}
        self._tok = self._log = None

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_pid"], d["_idx"], d["_tok"], d["_log"] = None, {}, None, None
        return d

    def _load(self):
        if self._pid != os.getpid():
            import pandas as pd
            df = pd.read_parquet(self.packed / "tokens.parquet", columns=["token", "log", "row"])
            rows = df.row.to_numpy(np.int64)
            if not (rows == np.arange(len(rows))).all():
                raise ValueError(f"{self.packed}/tokens.parquet: row column is not 0..N-1")
            self._tok = df.token.astype(str).to_numpy()
            self._log = df.log.astype(str).to_numpy()
            self._idx = {t: i for i, t in enumerate(self._tok.tolist())}
            self._pid = os.getpid()

    def __len__(self) -> int:
        self._load()
        return len(self._tok)

    def row_of(self, token: str) -> int:
        self._load()
        return int(self._idx.get(str(token), -1))

    def rows_of(self, tokens: Iterable[str]) -> np.ndarray:
        self._load()
        return np.asarray([self._idx.get(str(t), -1) for t in tokens], np.int64)

    def token_log(self, row: int) -> Tuple[str, str]:
        self._load()
        return str(self._tok[int(row)]), str(self._log[int(row)])

    @property
    def tokens(self) -> np.ndarray:
        self._load()
        return self._tok

    @property
    def logs(self) -> np.ndarray:
        self._load()
        return self._log


_ROWMAPS: Dict[str, RowMap] = {}


def _rowmap(packed: PathLike = PHASE1_DEFAULT["packed"]) -> RowMap:
    k = str(packed)
    if k not in _ROWMAPS:
        _ROWMAPS[k] = RowMap(k)
    return _ROWMAPS[k]


def row_of(token: str, packed: PathLike = PHASE1_DEFAULT["packed"]) -> int:
    """packed row of a token (-1 if absent)."""
    return _rowmap(packed).row_of(token)


def token_log(row: int, packed: PathLike = PHASE1_DEFAULT["packed"]) -> Tuple[str, str]:
    """(token, log) of a packed row (metric cache path = CM.mc_path('navtrain_train', log, token))."""
    return _rowmap(packed).token_log(row)


# ----------------------------------------------------------------------------------------------- Phase 1 arrays
class _Phase1:
    """Lazy per-process memmaps of the Phase 1 replay set (packed cand / ok, labels cand / kd_corr, kd_targets)."""

    FILES = {
        "cand": ("packed", "cand.npy"), "ok": ("packed", "ok.npy"),
        "y": ("labels_cand", "labels.npy"), "y_ok": ("labels_cand", "ok.npy"),
        "y_kdc": ("labels_kd_corr", "labels.npy"), "y_kdc_ok": ("labels_kd_corr", "ok.npy"),
        "kdc": ("kd_targets", "kd_corr_traj.npy"), "kd_prob": ("kd_targets", "kd_score_prob.npy"),
        "kd_prob_corr": ("kd_targets", "kd_score_prob_corr.npy"), "kd_c_lon": ("kd_targets", "kd_c_lon.npy"),
        "kd_e_lat": ("kd_targets", "kd_e_lat.npy"), "kd_ok": ("kd_targets", "kd_ok.npy"),
    }

    def __init__(self, phase1_cfg=None):
        self.paths = phase1_paths(phase1_cfg)
        self._pid, self._mm = None, {}

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_pid"], d["_mm"] = None, {}
        return d

    def check(self, n_rows: int = N_ROWS) -> None:
        """Shapes + kd_targets token order == packed (once, on open)."""
        import pandas as pd
        for name in self.FILES:
            a = self.arr(name)
            if a.shape[0] != n_rows:
                raise ValueError(f"phase1 {name}: {a.shape} rows != {n_rows}")
        kt = pd.read_parquet(self.paths["kd_targets"] / "tokens.parquet", columns=["token"]).token.astype(str)
        pt = pd.read_parquet(self.paths["packed"] / "tokens.parquet", columns=["token"]).token.astype(str)
        if len(kt) != len(pt) or not (kt.to_numpy() == pt.to_numpy()).all():
            raise ValueError(f"{self.paths['kd_targets']}: token order differs from packed")

    def arr(self, name: str) -> np.ndarray:
        if self._pid != os.getpid():
            self._mm, self._pid = {}, os.getpid()
        a = self._mm.get(name)
        if a is None:
            grp, fn = self.FILES[name]
            a = self._mm[name] = np.load(self.paths[grp] / fn, mmap_mode="r")
        return a

    def row_ok(self, r: int) -> bool:
        return bool(self.arr("ok")[r])

    def g_row(self, r: int, k: int = K) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Phase 1 G of a row = (traj [2k,8,3], y [2k,5] CK_KEYS, ok [2k]) = cat(cand, kd_corr)."""
        A = self.arr
        traj = np.concatenate([np.asarray(A("cand")[r, :k], np.float32), np.asarray(A("kdc")[r, :k], np.float32)])
        lab = np.concatenate([np.asarray(A("y")[r, :k], np.float32), np.asarray(A("y_kdc")[r, :k], np.float32)])
        y = lab[:, CK_IDX]
        ok = np.concatenate([np.asarray(A("y_ok")[r, :k], bool), np.asarray(A("y_kdc_ok")[r, :k], bool)])
        ok = ok & bool(A("ok")[r])
        return traj, y, ok


def phase1_label_prior(n_max: int = 20000, labels_dir: PathLike = PHASE1_DEFAULT["labels_cand"]):
    """Mean official label per CK key over ok candidates of <= n_max evenly spaced rows of the Phase 1 cand labels
    (= tools/ck/train_ck.label_prior('navtrain_train')); clipped to [1e-3, 1 - 1e-3]; None if absent."""
    d = Path(labels_dir)
    if not (d / "labels.npy").is_file():
        return None
    lab = np.load(d / "labels.npy", mmap_mode="r")
    ok = np.load(d / "ok.npy", mmap_mode="r")
    rows = np.unique(np.linspace(0, len(lab) - 1, min(n_max, len(lab))).astype(np.int64))
    L = np.asarray(lab[rows], np.float64)[..., CK_IDX]
    m = np.asarray(ok[rows], bool) & np.isfinite(L).all(-1)
    if not m.any():
        return None
    return np.clip(L[m].mean(0), 1e-3, 1 - 1e-3)


# ----------------------------------------------------------------------------------------------- target builder
try:
    from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder as _ATB
except Exception:   # pragma: no cover  (navsim always importable in the training env)
    _ATB = object


class CKE2ETargetBuilder(_ATB):
    """Per-token CK e2e targets (contract data.target_builder).  get_unique_name() = 'ck_e2e_targets'.

    Outputs (B dim added by the collate):
      ck_row int64 [] (-1 = token not in packed navtrain_train)
      ref_* (refiner.e2e.GTLoader surrogate GT) + ref_gt_ok bool []
      kd_bev_0 f16 [256,50,100] / kd_ok_0 bool [] (DET, BEVFusion t0); kd_bev_1 / kd_ok_1 (MAP, ReSMap; only with a
        map teacher run)
      ck_p1_ok bool [], ck_p1_cand f32 [16,8,3], ck_p1_y f32 [16,5], ck_p1_y_ok bool [16], ck_p1_kdc f32 [16,8,3],
      ck_p1_y_kdc f32 [16,5], ck_p1_y_kdc_ok bool [16], ck_p1_kd_prob f32 [16,5], ck_p1_kd_prob_corr f32 [16,5],
      ck_p1_kd_c_lon f32 [16,6], ck_p1_kd_e_lat f32 [16,6], ck_p1_kd_ok bool [16]
    Row ok flags are not folded into the per-candidate ok arrays (the loss does ck_p1_ok & ...).
    """

    P1_KEYS = ("cand", "y", "y_ok", "kdc", "y_kdc", "y_kdc_ok", "kd_prob", "kd_prob_corr", "kd_c_lon", "kd_e_lat",
               "kd_ok")

    def __init__(self, config=None, ck_cfg=None, ref_data_root: Optional[str] = None, k: int = K):
        try:
            super().__init__()
        except TypeError:   # pragma: no cover
            pass
        c = _cfg_dict(ck_cfg if ck_cfg is not None else config)
        self.k = int(c.get("topk", k) or k)
        if self.k != K:
            # Phase 1 replay arrays are K=16; a different topk would need a different replay set
            raise ValueError(f"CKE2ETargetBuilder: topk {self.k} != {K} (Phase 1 replay set is top-16)")
        det = c.get("teacher_det_run", TEACHER_DET_RUN)
        mp = c.get("teacher_map_run", TEACHER_MAP_RUN)
        self.teacher_runs: Tuple[str, ...] = tuple(str(r) for r in (det, mp) if r)
        if not det:
            raise ValueError("CKE2ETargetBuilder: teacher_det_run is required")
        root = ref_data_root or c.get("ref_data_root") or getattr(config, "ref_data_root", None) or REF_DATA_ROOT
        self.ref_data_root = str(root)
        self.p1 = _Phase1(c.get("phase1"))
        self.rows = RowMap(self.p1.paths["packed"])
        self._gtl = None
        self._pid = None
        self.n_calls = 0

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_gtl"], d["_pid"] = None, None
        return d

    def get_unique_name(self) -> str:
        return "ck_e2e_targets"

    def _loader(self):
        if self._gtl is None or self._pid != os.getpid():
            from navsim.agents.para_ssr.refiner.e2e import GTLoader
            self._gtl = GTLoader(self.ref_data_root, self.teacher_runs)
            self._pid = os.getpid()
        return self._gtl

    def compute_targets(self, scene) -> Dict[str, torch.Tensor]:
        return self.compute_for_token(str(scene.scene_metadata.initial_token))

    def phase1_row(self, r: int) -> Dict[str, torch.Tensor]:
        k = self.k
        out: Dict[str, np.ndarray] = {}
        ok = False
        if r >= 0:
            try:
                A = self.p1.arr
                ok = self.p1.row_ok(r)
                out["cand"] = np.array(A("cand")[r, :k], np.float32)
                out["y"] = np.array(A("y")[r, :k], np.float32)[:, CK_IDX]
                out["y_ok"] = np.array(A("y_ok")[r, :k], bool)
                out["kdc"] = np.array(A("kdc")[r, :k], np.float32)
                out["y_kdc"] = np.array(A("y_kdc")[r, :k], np.float32)[:, CK_IDX]
                out["y_kdc_ok"] = np.array(A("y_kdc_ok")[r, :k], bool)
                out["kd_prob"] = np.array(A("kd_prob")[r, :k], np.float32)
                out["kd_prob_corr"] = np.array(A("kd_prob_corr")[r, :k], np.float32)
                out["kd_c_lon"] = np.array(A("kd_c_lon")[r, :k], np.float32)
                out["kd_e_lat"] = np.array(A("kd_e_lat")[r, :k], np.float32)
                out["kd_ok"] = np.array(A("kd_ok")[r, :k], bool)
            except Exception:
                out, ok = {}, False
        if not out:
            out = self._p1_empty()
            ok = False
        res = {f"ck_p1_{n}": torch.as_tensor(np.ascontiguousarray(out[n])) for n in self.P1_KEYS}
        res["ck_p1_ok"] = torch.tensor(bool(ok))
        return res

    def _p1_empty(self) -> Dict[str, np.ndarray]:
        k = self.k
        return {"cand": np.zeros((k, 8, 3), np.float32), "y": np.zeros((k, 5), np.float32),
                "y_ok": np.zeros(k, bool), "kdc": np.zeros((k, 8, 3), np.float32),
                "y_kdc": np.zeros((k, 5), np.float32), "y_kdc_ok": np.zeros(k, bool),
                "kd_prob": np.zeros((k, 5), np.float32), "kd_prob_corr": np.zeros((k, 5), np.float32),
                "kd_c_lon": np.zeros((k, 6), np.float32), "kd_e_lat": np.zeros((k, 6), np.float32),
                "kd_ok": np.zeros(k, bool)}

    def compute_for_token(self, token: str) -> Dict[str, torch.Tensor]:
        self.n_calls += 1
        try:
            r = self.rows.row_of(token)
        except Exception:
            r = -1
        out: Dict[str, torch.Tensor] = {"ck_row": torch.tensor(int(r), dtype=torch.int64)}
        out.update(self._loader().load(str(token)))      # ref_* + kd_bev_i / kd_ok_i (GTLoader rules)
        out.update(self.phase1_row(r))
        return out


# ----------------------------------------------------------------------------------------------- record chunks
def new_attempt_id() -> str:
    """8 hex chars, new per process and epoch start."""
    return secrets.token_hex(4)


def rec_rank_dir(io_dir: PathLike, epoch: int, rank: int) -> Path:
    return Path(io_dir) / "rec" / f"ep{int(epoch):03d}" / f"r{int(rank)}"


def rec_chunk_path(io_dir: PathLike, epoch: int, rank: int, attempt: str, seq: int) -> Path:
    attempt = str(attempt)
    if not re.fullmatch(r"[0-9a-f]{8}", attempt):
        raise ValueError(f"attempt id {attempt!r} is not 8 hex chars")
    return rec_rank_dir(io_dir, epoch, rank) / f"c_{attempt}_{int(seq):06d}.npz"


def parse_rec_chunk_path(path: PathLike) -> Optional[Dict[str, Any]]:
    """{epoch, rank, attempt, seq, name, path} for a normal chunk name, else None (tmp / other files)."""
    p = Path(path)
    m = _CHUNK_RE.match(p.name)
    me, mr = _EP_RE.match(p.parent.parent.name), _RANK_RE.match(p.parent.name)
    if not (m and me and mr):
        return None
    return {"epoch": int(me.group(1)), "rank": int(mr.group(1)), "attempt": m.group(1), "seq": int(m.group(2)),
            "name": p.name, "path": p}


def _rec_arrays(row, cand, cand_idx, kd_corr, kd_ok, gstep, epoch, rank) -> Dict[str, np.ndarray]:
    def _np(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    row = _np(row).astype(np.int64).reshape(-1)
    n = len(row)
    a = {"row": row,
         "cand": _np(cand).astype(np.float32),
         "cand_idx": _np(cand_idx).astype(np.int16),
         "kd_corr": _np(kd_corr).astype(np.float32),
         "kd_ok": _np(kd_ok).astype(bool),
         "gstep": np.broadcast_to(_np(gstep).astype(np.int64).reshape(-1), (n,)).copy(),
         "epoch": np.asarray(int(epoch), np.int32), "rank": np.asarray(int(rank), np.int32)}
    _check_rec(a)
    return a


def _check_rec(a: Dict[str, np.ndarray]) -> None:
    n = a["row"].shape[0]
    k = a["cand"].shape[1] if a["cand"].ndim == 4 else -1
    want = {"row": ((n,), np.int64), "cand": ((n, k, 8, 3), np.float32), "cand_idx": ((n, k), np.int16),
            "kd_corr": ((n, k, 8, 3), np.float32), "kd_ok": ((n, k), np.bool_), "gstep": ((n,), np.int64),
            "epoch": ((), np.int32), "rank": ((), np.int32)}
    for name, (shape, dt) in want.items():
        if name not in a:
            raise ValueError(f"rec chunk: missing {name}")
        x = a[name]
        if tuple(x.shape) != shape or x.dtype != dt:
            raise ValueError(f"rec chunk {name}: {x.shape} {x.dtype} != {shape} {np.dtype(dt)}")


def write_rec_chunk(path: PathLike, row, cand, cand_idx, kd_corr, kd_ok, gstep, epoch: int, rank: int) -> Path:
    """Atomic: np.savez to '.{name}.tmp{pid}' in the same dir, then os.replace.  Arrays: row int64 [n],
    cand f32 [n,K,8,3], cand_idx int16 [n,K], kd_corr f32 [n,K,8,3], kd_ok bool [n,K], gstep int64 [n] (a scalar is
    broadcast), epoch int32 [], rank int32 []."""
    path = Path(path)
    a = _rec_arrays(row, cand, cand_idx, kd_corr, kd_ok, gstep, epoch, rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        np.savez(f, **a)
    os.replace(tmp, path)
    return path


def read_rec_chunk(path: PathLike) -> Dict[str, Any]:
    """-> dict of the arrays (shape / dtype checked); epoch and rank as python ints too ('epoch_i', 'rank_i')."""
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    _check_rec(a)
    a["epoch_i"], a["rank_i"] = int(a["epoch"]), int(a["rank"])
    return a


def write_rec_done(io_dir: PathLike, epoch: int, rank: int, world_size: int, attempt: str, chunks: Sequence[str],
                   n_tokens: int) -> Path:
    """{io_dir}/rec/ep{E:03d}/r{R}/DONE_{attempt}.json (atomic; only at a normal epoch end).  chunks = chunk file
    names (or paths) of this attempt."""
    p = rec_rank_dir(io_dir, epoch, rank) / f"DONE_{attempt}.json"
    _atomic_write_json(p, {"epoch": int(epoch), "rank": int(rank), "world_size": int(world_size),
                           "attempt": str(attempt), "chunks": [Path(c).name for c in chunks],
                           "n_tokens": int(n_tokens), "time": time.time(),
                           "utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())})
    return p


def list_rec_chunks(io_dir: PathLike, epoch: Optional[int] = None) -> List[Dict[str, Any]]:
    """Normal chunk files (tmp names ignored), sorted by (epoch, mtime, rank, name)."""
    base = Path(io_dir) / "rec"
    if not base.is_dir():
        return []
    eps = [base / f"ep{int(epoch):03d}"] if epoch is not None else sorted(base.glob("ep[0-9][0-9][0-9]"))
    out = []
    for ed in eps:
        for p in ed.glob("r*/c_*.npz"):
            d = parse_rec_chunk_path(p)
            if d is None:
                continue
            try:
                d["mtime"] = p.stat().st_mtime
            except FileNotFoundError:
                continue
            out.append(d)
    out.sort(key=lambda d: (d["epoch"], d["mtime"], d["rank"], d["name"]))
    return out


def read_rec_done(io_dir: PathLike, epoch: int) -> Dict[int, List[Dict[str, Any]]]:
    """rank -> list of DONE json dicts of epoch (every attempt that finished normally)."""
    out: Dict[int, List[Dict[str, Any]]] = {}
    ed = Path(io_dir) / "rec" / f"ep{int(epoch):03d}"
    for p in sorted(ed.glob("r*/DONE_*.json")):
        mr = _RANK_RE.match(p.parent.name)
        if not mr:
            continue
        try:
            out.setdefault(int(mr.group(1)), []).append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    return out


# ----------------------------------------------------------------------------------------------- generations
GEN_FILES = {"traj": ((G_K, 8, 3), np.float32, np.nan), "labels": ((G_K, N_LAB), np.float32, np.nan),
             "cand_ok": ((G_K,), np.bool_, False), "row_state": ((), np.uint8, 0)}


def gen_dir(io_dir: PathLike, epoch: int) -> Path:
    return Path(io_dir) / "lab" / f"ep{int(epoch):03d}"


def list_generations(io_dir: PathLike) -> List[int]:
    """Epochs whose generation is ready (meta.json present), ascending."""
    base = Path(io_dir) / "lab"
    if not base.is_dir():
        return []
    out = []
    for d in base.glob("ep[0-9][0-9][0-9]"):
        m = _EP_RE.match(d.name)
        if m and (d / "meta.json").is_file():
            out.append(int(m.group(1)))
    return sorted(out)


def open_generation(io_dir: PathLike, epoch: int, n_rows: int = N_ROWS, mode: str = "r") -> Dict[str, Any]:
    """Generation memmaps {traj f32 [N,32,8,3], labels f32 [N,32,9] (LABEL_COLS), cand_ok bool [N,32],
    row_state uint8 [N], meta, dir}.
      'w+' : labeler; creates the files under a FileLock (NaN / False / 0 filled, flushed; meta.json written LAST =
             ready marker).  If the generation already exists it is opened 'r+' (idempotent restart).
      'r+' : labeler restart (must exist).   'r' : reader (must exist)."""
    if mode not in ("w+", "r+", "r"):
        raise ValueError(f"open_generation mode {mode!r}")
    d = gen_dir(io_dir, epoch)
    meta_p = d / "meta.json"
    if mode == "w+":
        with _FileLock(d / ".lock"):
            if not meta_p.is_file():
                d.mkdir(parents=True, exist_ok=True)
                for name, (shape, dt, fill) in GEN_FILES.items():
                    mm = np.lib.format.open_memmap(d / f"{name}.npy", mode="w+", dtype=dt, shape=(int(n_rows),) + shape)
                    mm[...] = fill
                    mm.flush()
                    del mm
                _atomic_write_json(meta_p, {"epoch": int(epoch), "n_rows": int(n_rows), "k": G_K,
                                            "order": "cat(recorded cand 16, recorded tau'_KD 16)",
                                            "label_cols": list(LABEL_COLS),
                                            "files": {n: [list(s), np.dtype(t).name] for n, (s, t, _) in
                                                      GEN_FILES.items()},
                                            "row_state": "0 none, 1 done (written after traj/labels/cand_ok)",
                                            "created": time.time(),
                                            "utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())})
        mode = "r+"
    if not meta_p.is_file():
        raise FileNotFoundError(f"generation {d} not ready (no meta.json)")
    meta = json.loads(meta_p.read_text())
    if int(meta["n_rows"]) != int(n_rows):
        raise ValueError(f"{d}: n_rows {meta['n_rows']} != {n_rows}")
    out: Dict[str, Any] = {"meta": meta, "dir": d}
    for name, (shape, dt, _) in GEN_FILES.items():
        a = np.load(d / f"{name}.npy", mmap_mode=mode)
        if a.shape != (int(n_rows),) + shape or a.dtype != dt:
            raise ValueError(f"{d}/{name}.npy: {a.shape} {a.dtype}")
        out[name] = a
    return out


def write_generation_rows(gen: Dict[str, Any], rows, traj, labels, cand_ok, overwrite: bool = False) -> np.ndarray:
    """Single-writer helper for the labeler: for rows not yet done (row_state != 1, unless overwrite) write traj
    [n,32,8,3], labels [n,32,9], cand_ok [n,32]; flush; then row_state = 1; flush.  Duplicate rows inside one call:
    the first occurrence wins.  -> bool [n] mask of the entries actually written."""
    rows = np.asarray(rows, np.int64).reshape(-1)
    traj, labels, cand_ok = np.asarray(traj, np.float32), np.asarray(labels, np.float32), np.asarray(cand_ok, bool)
    n = len(rows)
    assert traj.shape == (n, G_K, 8, 3) and labels.shape == (n, G_K, N_LAB) and cand_ok.shape == (n, G_K), \
        (traj.shape, labels.shape, cand_ok.shape)
    st = gen["row_state"]
    _, first = np.unique(rows, return_index=True)
    w = np.zeros(n, bool)
    w[first] = True
    if not overwrite:
        w &= np.asarray(st[rows]) != 1
    if not w.any():
        return w
    r = rows[w]
    order = np.argsort(r)                                   # sorted fancy writes
    r, sel = r[order], np.flatnonzero(w)[order]
    gen["traj"][r] = traj[sel]
    gen["labels"][r] = labels[sel]
    gen["cand_ok"][r] = cand_ok[sel]
    for name in ("traj", "labels", "cand_ok"):
        gen[name].flush()
    st[r] = 1
    st.flush()
    return w


# ----------------------------------------------------------------------------------------------- label store
class LabelStore:
    """G (GT BCE trajectory set) per packed row for the on-policy epochs (contract data.label_store).

    refresh(max_epoch) -> stats: per row src_epoch = the largest generation E <= max_epoch with row_state == 1
      (-1 = none -> Phase 1 set).  Pure read; the epoch-e caller passes max_epoch = e - label_lag (never e itself).
    lookup(rows [B]) -> G_traj f32 [B,32,8,3], G_y f32 [B,32,5] (CK_KEYS), G_ok bool [B,32] (cand_ok & finite),
      G_src int8 [B] (0 Phase 1, 1 the max_epoch generation, 2 an older one), G_epoch int16 [B] (-1 Phase 1).
      row < 0 -> G_ok all False (G_src 0, G_epoch -1).  Non-finite values are zeroed (ok False).
    Files are opened lazily per process; pickling drops them and the src index (re-read on first use after a pid
    change, with the same max_epoch).  No collectives.
    """

    def __init__(self, io_dir: PathLike, phase1_cfg=None, n_rows: int = N_ROWS, k: int = K):
        self.io_dir = Path(io_dir)
        self.n_rows, self.k = int(n_rows), int(k)
        self.p1 = _Phase1(phase1_cfg)
        self.max_epoch: Optional[int] = None
        self.src_epoch: Optional[np.ndarray] = None
        self.last_stats: Dict[str, Any] = {}
        self._gens: Dict[int, Dict[str, Any]] = {}
        self._pid = None
        self.cum = {"lookup_rows": 0, "src_phase1": 0, "src_prev": 0, "src_older": 0, "no_row": 0}

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
            g = self._gens[epoch] = open_generation(self.io_dir, epoch, self.n_rows, "r")
        return g

    def refresh(self, max_epoch: int) -> Dict[str, Any]:
        t0 = time.time()
        if self._pid != os.getpid():
            self._gens, self._pid = {}, os.getpid()
        max_epoch = int(max_epoch)
        src = np.full(self.n_rows, -1, np.int16)
        gens = [e for e in list_generations(self.io_dir) if e <= max_epoch]
        used = []
        for e in sorted(gens, reverse=True):
            try:
                st = np.array(np.load(gen_dir(self.io_dir, e) / "row_state.npy", mmap_mode="r"))
            except (OSError, ValueError):
                continue
            if st.shape != (self.n_rows,):
                raise ValueError(f"generation ep{e:03d}: row_state {st.shape} != ({self.n_rows},)")
            m = (src < 0) & (st == 1)
            src[m] = e
            used.append(e)
            self._gens.pop(e, None)       # reopen lazily (a fresh memmap view; cheap)
        self.max_epoch, self.src_epoch = max_epoch, src
        n_prev = int(((src == max_epoch) & (src >= 0)).sum()) if max_epoch >= 0 else 0   # src -1 = no generation
        n_any = int((src >= 0).sum())
        lag = {}
        for e in used:
            c = int((src == e).sum())
            if c:
                lag[str(max_epoch - e + 1)] = c          # label lag in epochs for a consumer at epoch max_epoch + 1
        st = {"max_epoch": max_epoch, "gens": sorted(used), "n_rows": self.n_rows, "n_prev": n_prev,
              "n_older": n_any - n_prev, "n_phase1": self.n_rows - n_any,
              "frac_prev": n_prev / self.n_rows, "frac_older": (n_any - n_prev) / self.n_rows,
              "frac_phase1": (self.n_rows - n_any) / self.n_rows, "lag_counts": lag,
              "sec": round(time.time() - t0, 4)}
        self.last_stats = st
        return st

    def lookup(self, rows, device=None) -> Dict[str, torch.Tensor]:
        self._proc()
        r = (rows.detach().cpu().numpy() if torch.is_tensor(rows) else np.asarray(rows)).astype(np.int64).reshape(-1)
        B, K2 = len(r), 2 * self.k
        traj = np.zeros((B, K2, 8, 3), np.float32)
        y = np.zeros((B, K2, 5), np.float32)
        ok = np.zeros((B, K2), bool)
        src = np.zeros(B, np.int8)
        gep = np.full(B, -1, np.int16)
        se = self.src_epoch
        for b, rr in enumerate(r):
            if rr < 0 or rr >= self.n_rows:
                self.cum["no_row"] += 1
                continue
            e = int(se[rr]) if se is not None else -1
            if e >= 0:
                g = self._gen(e)
                t = np.asarray(g["traj"][rr], np.float32)
                lab = np.asarray(g["labels"][rr], np.float32)[:, CK_IDX]
                o = np.asarray(g["cand_ok"][rr], bool)
                src[b], gep[b] = (SRC_PREV if e == self.max_epoch else SRC_OLDER), e
            else:
                t, lab, o = self.p1.g_row(int(rr), self.k)
                src[b] = SRC_PHASE1
            fin_t = np.isfinite(t).all((-1, -2))
            fin_y = np.isfinite(lab).all(-1)
            o = o & fin_t & fin_y
            traj[b] = np.where(fin_t[:, None, None], t, 0.0)
            y[b] = np.where(fin_y[:, None], lab, 0.0)
            ok[b] = o
            self.cum["lookup_rows"] += 1
            self.cum[("src_phase1", "src_prev", "src_older")[int(src[b])]] += 1
        out = {"G_traj": torch.from_numpy(traj), "G_y": torch.from_numpy(y), "G_ok": torch.from_numpy(ok),
               "G_src": torch.from_numpy(src), "G_epoch": torch.from_numpy(gep)}
        if device is not None:
            out = {k: v.to(device, non_blocking=True) for k, v in out.items()}
        return out

    def stats(self, reset: bool = False) -> Dict[str, Any]:
        """Cumulative lookup source counts since the last reset (+ last refresh stats)."""
        c = dict(self.cum)
        n = max(c["lookup_rows"], 1)
        c.update({"frac_phase1": c["src_phase1"] / n, "frac_prev": c["src_prev"] / n,
                  "frac_older": c["src_older"] / n, "refresh": dict(self.last_stats)})
        if reset:
            self.cum = {k: 0 for k in self.cum}
        return c
