"""refiner-v3 object pack reader + training augmentation (SPEC 3.4, 5.1; owner B1).

Pack layout (written once on CPU by tools/ck/v3/build_obj_pack.py, shared by every arm):
  P = <root>/<source>/<split>/   root default D3/objpack, source 'bevfusion', split navtrain_train | navtrain_val | navtest
  row r = packed row r of CK_DATA/packed/<split>/tokens.parquet
  tokens.parquet (token, row) | obj.npy f32 [N, 3, 64, 17] | obj_mask.npy bool [N, 3, 64] | frame_valid.npy bool [N, 3]
  frame_dt.npy f32 [N, 3] | frame_src.npy i8 [N, 3] | n_raw.npy i16 [N, 3] | obj_ok.npy bool [N]
  fc_xyh.npy f32 [N, 64, 10, 3] | fc_state.npy i8 [N, 64, 10] | fc_track.npy i32 [N, 64]   (train / val only)
  extras (audit only, never model inputs): frame_why.npy i8 [N, 3], frame_token.npy <U16 [N, 3]
  build.json (version v3_objpack_1, provenance, counts, per-file sha256)

``ObjPack.item(row)`` -> obj_in, obj_mask, frame_valid, obj_ok (+ fc_xyh, fc_state when targets=True).  Only obj_in,
obj_mask and frame_valid (config.OBJ_INPUT_KEYS) may reach the model; fc_* are loss / metric targets.

``augment_objects`` (training only, after target matching, identical for arms a, b, d): rng =
token_rng(token, epoch, seed, "objaug"); draws in this order, always all of them:
  (1) U(0,1) x 2 for past frames 1, 2: u < p_frame_drop -> frame_valid False, mask False;
  (2) U(0,1) [3, 64]: u < p_obj_drop -> mask False;
  (3) N(0, v_noise^2) [3, 64, 2] added to cols 6, 7 where mask; rows zeroed where the mask is False.
Targets are untouched.  cfg off -> the item is returned unchanged and nothing is drawn.
Basis [실측 SPEC 5.1]: in-sample velocity error median 0.27 m/s (48 moving vehicles); navtest recall .846.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from navsim.agents.para_ssr.ck.anchor_sampler import token_rng
from navsim.agents.para_ssr.ck.constants import CK_DATA
from navsim.agents.para_ssr.ck.temporal.config import (D_OBJ, FC_STEPS, K_OBJ, N_FRAMES, PACK_ROOT, SOURCES,
                                                       TemporalCfg)

PACK_VERSION = "v3_objpack_1"
AUG_STREAM = "objaug"
SPLITS = ("navtrain_train", "navtrain_val", "navtest")
C_VX, C_VY = 6, 7

# name -> (dtype, trailing shape); the first dimension is N
PACK_FILES = {
    "obj": (np.float32, (N_FRAMES, K_OBJ, D_OBJ)),
    "obj_mask": (np.bool_, (N_FRAMES, K_OBJ)),
    "frame_valid": (np.bool_, (N_FRAMES,)),
    "frame_dt": (np.float32, (N_FRAMES,)),
    "frame_src": (np.int8, (N_FRAMES,)),
    "n_raw": (np.int16, (N_FRAMES,)),
    "obj_ok": (np.bool_, ()),
    "frame_why": (np.int8, (N_FRAMES,)),
    "frame_token": (np.dtype("<U16"), (N_FRAMES,)),
}
TARGET_FILES = {
    "fc_xyh": (np.float32, (K_OBJ, FC_STEPS, 3)),
    "fc_state": (np.int8, (K_OBJ, FC_STEPS)),
    "fc_track": (np.int32, (K_OBJ,)),
}


def pack_dir(source: str, split: str, root: Optional[Union[str, Path]] = None) -> Path:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    return Path(root if root is not None else PACK_ROOT) / source / split


def packed_tokens(split: str, packed_root: Optional[Union[str, Path]] = None) -> np.ndarray:
    """token order of CK_DATA/packed/<split>/tokens.parquet (the row order every CK dataset uses)."""
    import pandas as pd
    p = Path(packed_root if packed_root is not None else Path(CK_DATA) / "packed") / split / "tokens.parquet"
    return pd.read_parquet(p, columns=["token"]).token.astype(str).to_numpy()


class ObjPack:
    """Read-only object pack of one (source, split).  Checks the version, the file shapes and the token order
    (``expect_tokens``, default: the first N tokens of CK_DATA/packed/<split>/tokens.parquet).  Arrays are memory
    mapped lazily once per process (fork-safe, picklable: mmaps are dropped on pickling)."""

    def __init__(self, split: str, source: str = "bevfusion", root: Optional[Union[str, Path]] = None,
                 targets: bool = True, expect_tokens: Optional[np.ndarray] = None):
        import pandas as pd

        self.split, self.source = str(split), str(source)
        self.dir = pack_dir(source, split, root)
        bj = self.dir / "build.json"
        if not bj.is_file():
            raise FileNotFoundError(f"object pack without build.json: {bj}")
        self.build = json.loads(bj.read_text())
        if self.build.get("version") != PACK_VERSION:
            raise ValueError(f"{self.dir}: pack version {self.build.get('version')!r} != {PACK_VERSION!r}")
        if self.build.get("source") != self.source or self.build.get("split") != self.split:
            raise ValueError(f"{self.dir}: build.json source/split {self.build.get('source')}/{self.build.get('split')}"
                             f" != {self.source}/{self.split}")
        if self.build.get("status") != "complete":
            raise ValueError(f"{self.dir}: build status {self.build.get('status')!r} (incomplete pack)")
        tdf = pd.read_parquet(self.dir / "tokens.parquet")
        self.tokens = tdf.token.astype(str).to_numpy()
        self.n = len(self.tokens)
        if not np.array_equal(tdf["row"].to_numpy(), np.arange(self.n)):
            raise ValueError(f"{self.dir}: tokens.parquet rows are not 0..N-1")
        exp = packed_tokens(self.split)[: self.n] if expect_tokens is None else np.asarray(expect_tokens).astype(str)
        if len(exp) != self.n or not np.array_equal(exp, self.tokens):
            raise ValueError(f"{self.dir}: token order differs from the expected packed order "
                             f"({self.n} pack rows vs {len(exp)} expected)")
        self.targets = bool(targets)
        if self.targets and not self.has_targets:
            raise ValueError(f"{self.dir}: targets requested but the pack has none (navtest / --no-targets build)")
        self._names = list(PACK_FILES) + (list(TARGET_FILES) if self.targets else [])
        self._pid = None
        self._mm: Dict[str, np.ndarray] = {}
        m = self._arrays()
        for name, (dt, shp) in {**PACK_FILES, **(TARGET_FILES if self.targets else {})}.items():
            a = m[name]
            if a.shape != (self.n,) + tuple(shp) or a.dtype != np.dtype(dt):
                raise ValueError(f"{self.dir}/{name}.npy: {a.shape} {a.dtype} != {(self.n,) + tuple(shp)} {np.dtype(dt)}")

    @property
    def has_targets(self) -> bool:
        return bool(self.build.get("targets")) and all((self.dir / f"{k}.npy").is_file() for k in TARGET_FILES)

    def __len__(self) -> int:
        return self.n

    def _arrays(self) -> Dict[str, np.ndarray]:
        pid = os.getpid()
        if self._pid != pid:
            self._mm = {k: np.load(self.dir / f"{k}.npy", mmap_mode="r") for k in self._names}
            self._pid = pid
        return self._mm

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_mm"], d["_pid"] = {}, None
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def item(self, row: int) -> Dict[str, np.ndarray]:
        """-> obj_in f32 [3, 64, 17], obj_mask bool [3, 64], frame_valid bool [3], obj_ok bool [] (+ fc_xyh f32
        [64, 10, 3], fc_state i8 [64, 10] if targets).  Fresh arrays (safe to modify)."""
        r = int(row)
        if not 0 <= r < self.n:
            raise IndexError(f"{self.dir}: row {r} outside 0..{self.n - 1}")
        m = self._arrays()
        out = dict(obj_in=np.array(m["obj"][r], np.float32), obj_mask=np.array(m["obj_mask"][r], bool),
                   frame_valid=np.array(m["frame_valid"][r], bool), obj_ok=np.array(bool(m["obj_ok"][r])))
        if self.targets:
            out["fc_xyh"] = np.array(m["fc_xyh"][r], np.float32)
            out["fc_state"] = np.array(m["fc_state"][r], np.int8)
        return out

    def token(self, row: int) -> str:
        return str(self.tokens[int(row)])


def _as_cfg(cfg) -> TemporalCfg:
    if isinstance(cfg, TemporalCfg):
        return cfg
    if cfg is None or isinstance(cfg, dict):
        return TemporalCfg.from_dict(cfg)
    raise TypeError(f"cfg must be a TemporalCfg or its dict, got {type(cfg)}")


def augment_objects(it: Dict[str, np.ndarray], token: str, epoch: int, seed: int, cfg) -> Dict[str, np.ndarray]:
    """Training-time detection-noise augmentation (module docstring).  Returns a new dict (input arrays are not
    modified); cfg off -> ``it`` itself, no RNG draw."""
    cfg = _as_cfg(cfg)
    if not cfg.on:
        return it
    rng = token_rng(token, epoch, seed, AUG_STREAM)
    u_frame = rng.random(2)                                            # (1) past frames 1, 2
    u_obj = rng.random((N_FRAMES, K_OBJ))                              # (2) object slots, every frame
    noise = rng.standard_normal((N_FRAMES, K_OBJ, 2)) * float(cfg.v_noise)   # (3) velocity noise (= N(0, v^2))
    out = dict(it)
    fv = np.array(it["frame_valid"], bool, copy=True)
    mask = np.array(it["obj_mask"], bool, copy=True)
    obj = np.array(it["obj_in"], np.float32, copy=True)
    if fv.shape != (N_FRAMES,) or mask.shape != (N_FRAMES, K_OBJ) or obj.shape != (N_FRAMES, K_OBJ, D_OBJ):
        raise ValueError(f"augment_objects: shapes {obj.shape} {mask.shape} {fv.shape}")
    for i, f in enumerate((1, 2)):
        if u_frame[i] < float(cfg.p_frame_drop):
            fv[f] = False
            mask[f] = False
    mask &= ~(u_obj < float(cfg.p_obj_drop))
    obj[..., C_VX:C_VY + 1] += np.where(mask[..., None], noise, 0.0).astype(np.float32)
    obj[~mask] = 0.0
    out["obj_in"], out["obj_mask"], out["frame_valid"] = obj, mask, fv
    return out


__all__ = ["PACK_VERSION", "AUG_STREAM", "SPLITS", "PACK_FILES", "TARGET_FILES", "pack_dir", "packed_tokens",
           "ObjPack", "augment_objects"]
