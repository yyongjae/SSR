"""CK2 teacher dataset: one item = one navtrain token with K = 32 sampled raw anchors + a subset of rule variants.

CK2Dataset(split, bev in {'T', 'M', 'none'}, n_var=32, var_name='var_separate_sampler16', gt=False, epoch=0, seed=0,
           sampler_seed=0, rows=None, limit=0, root=None, ck2_root=None, ep_target='official')

  root       CK data root (default CK_DATA, env CK_DATA_ROOT honoured): packed/<split>/{tokens.parquet, gt_traj.npy,
             status.npy, ok.npy} and labels/<split>/raw256/{labels.npy [N, 256, 9], ok.npy [N, 256]} (official
             per-trajectory labels of all 256 raw anchors; row = packed row).
  ck2_root   default <root>/ck2: labels/<split>/<var_name>/{traj.npy [N, 16*6, 8, 3], labels.npy [N, 96, 9],
             ok.npy [N, 96], index.npz} (tools/ck/data/label_variants.py; column c = k * V + v, k = anchor slot of the
             fixed near+mid-16 subset in GT-rank order, v = variant (0 id, 1 a-1.0, 2 a-0.5, 3 a+0.5, 4 l-0.5,
             5 l+0.5)).  n_var = 0 -> no variant keys.
  bev        T: refiner.data.TeacherCache (BEVFusion t0), M: refiner.resmap_cache.ResmapCache (ReSMap; no
             navtrain_val cache), none: no BEV key (tests / label-only use).  Unreadable BEV -> zeros + bev_ok False.
  gt         GTLoader surrogate GT (ref_* keys), training only.
  epoch      sampler epoch (set_epoch(e) before building the epoch's DataLoader; workers fork the current value).
  ep_target  'official' (default: y / var_y EP = the official label column, old behaviour bit-identical) | 'decoupled'
             (EP = clip(raw_progress / max(pdm_progress_eff, raw_progress), 0, 1) if max > 5 m else 1, i.e. official EP
             without the NC * DAC * DDC factor; user 2026-10-08): navsim.agents.para_ssr.ck.ep_target.ck_targets turns
             every 9-column label row into the 5 CK targets (anchors, variants, ck2_label_prior).  y_pdms / var_pdms
             stay OFFICIAL; the AnchorSampler draw always sees the official labels (pass mask NC / DAC / TTC / C).

Per item (row r, token t, epoch e)
  anchors   AnchorSampler(seed=sampler_seed).sample(t, gt_traj[r], raw256 labels[r] (NC, DAC, EP, TTC, C), epoch=e,
            shuffle=True): 8 near + 8 mid + 16 pass/fail-stratified far anchors in a random slot order.
  variants  the valid (index.npz 'valid'), non-identity columns of the token's 80 = 16 x 5 variant candidates, a random
            subset of n_var of them in a random order: stream token_rng(t, e, seed, 'var').  Invalid columns (lateral
            variant of an anchor with S_8 < 3 m = duplicate of its speed-only parent) are never drawn; measured min
            valid per token = 66 (train and val), so n_var <= 66 never pads (padding = repeated columns with
            var_ok False).
  Determinism: everything is a function of (seed, sampler_seed, epoch, token) -- independent of worker / rank / call
  order (anchor_sampler.token_rng = blake2b-keyed PCG64).

Item keys
  model inputs (MODEL_INPUT_KEYS; nothing else may reach the network)
    bev f16 [256, 50, 100] (+ bev_ok), cand f32 [K, 8, 3] (raw anchors), status f32 [8], var_traj f32 [V, 8, 3]
  labels (CK_KEYS order nc, dac, ep, ttc, comfort; EP per ep_target; ok = label ok & all 5 targets finite)
    y f32 [K, 5], y_ok bool [K], y_pdms f32 [K] (official); var_y f32 [V, 5], var_ok bool [V], var_pdms f32 [V]
  training-time metadata (GT- / label-derived: never model inputs)
    anc_idx int64 [K] anchor ids, anc_group int8 [K] (0 near / 1 mid / 2 strat), anc_pi f64 [K] inclusion
    probability, anc_w f32 [K] = 1 / pi (Sum = 256), var_anchor int64 [V] parent anchor id, var_v int8 [V] variant
    index 1..5, var_col int64 [V] column in the variant files, var_pi f32 [V] = n_var / n_valid (inclusion
    probability of a valid variant), gt_traj f32 [8, 3] (surrogate human-overlap mask only), ref_* (gt=True).
collate: tools.ck.data.ck_dataset.collate_ck (stacks every key; 'tokens' list, 'rows' int64).
Arrays are memory-mapped lazily per process (fork or spawn workers).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root of this worktree
from tools.ck.data import common as CM  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from navsim.agents.para_ssr.ck import anchor_sampler as AS  # noqa: E402
from navsim.agents.para_ssr.ck.ep_target import check_ep_target, ck_targets  # noqa: E402

BEV_SHAPE = (256, 50, 100)
CK_LABEL_IDX = tuple(CM.C.CK_LABEL_IDX)
PDMS_IDX = CM.LABEL_COLS.index("pdms")
VAR_NAME = "var_separate_sampler16"
RAW_LABELS = "raw256"
MODEL_INPUT_KEYS = ("bev", "cand", "status", "var_traj")
VAR_STREAM = "var"


def ck2_dir(root=None, ck2_root=None) -> Path:
    return Path(ck2_root) if ck2_root else CM.root_dir(root) / "ck2"


def var_dir(split: str, var_name: str = VAR_NAME, root=None, ck2_root=None) -> Path:
    return ck2_dir(root, ck2_root) / "labels" / split / var_name


class CK2Dataset(torch.utils.data.Dataset):
    def __init__(self, split: str, bev: str = "T", n_var: int = 32, var_name: str = VAR_NAME, gt: bool = False,
                 epoch: int = 0, seed: int = 0, sampler_seed: int = 0, rows: Optional[np.ndarray] = None,
                 limit: int = 0, root=None, ck2_root=None, anchors: Optional[np.ndarray] = None,
                 ep_target: str = "official"):
        assert bev in ("T", "M", "none"), bev
        self.split, self.bev, self.n_var = split, bev, int(n_var)
        self.ep_target = check_ep_target(ep_target)
        self.sp = CM.split(split)
        if bev == "M" and not self.sp.get("map_teacher", False):
            raise ValueError(f"no ReSMap teacher cache for {split} (map_teacher False)")
        self.root = CM.root_dir(root)
        self.pk = CM.packed_dir(split, self.root)
        self.tokens = pd.read_parquet(self.pk / "tokens.parquet", columns=["token"]).token.astype(str).to_numpy()
        self.n = len(self.tokens)
        self.epoch, self.seed, self.sampler_seed = int(epoch), int(seed), int(sampler_seed)
        self.sampler = AS.AnchorSampler(anchors=AS.load_anchors() if anchors is None else anchors,
                                        seed=self.sampler_seed)
        self.K = self.sampler.K
        rl = CM.labels_dir(split, RAW_LABELS, self.root)
        self._paths: Dict[str, Path] = {"status": self.pk / "status.npy", "gt_traj": self.pk / "gt_traj.npy",
                                        "raw_labels": rl / "labels.npy", "raw_ok": rl / "ok.npy"}
        rs = np.load(self._paths["raw_labels"], mmap_mode="r").shape
        if rs != (self.n, AS.N_ANCHORS, len(CM.LABEL_COLS)):
            raise ValueError(f"{rl}/labels.npy {rs} != ({self.n}, {AS.N_ANCHORS}, {len(CM.LABEL_COLS)})")
        self.var_dir, self.var_meta = None, None
        keep = np.load(self.pk / "ok.npy").astype(bool)
        keep &= np.isfinite(np.load(self._paths["gt_traj"], mmap_mode="r")).all((1, 2))
        rok = np.load(self._paths["raw_ok"], mmap_mode="r")
        keep &= np.asarray(rok).all(1)
        if self.n_var > 0:
            vd = var_dir(split, var_name, self.root, ck2_root)
            self.var_dir = vd
            b = json.loads((vd / "build.json").read_text())
            self.var_meta = {k: b.get(k) for k in ("version", "variants_version", "combine", "subset", "n_anchors",
                                                    "epoch", "seed", "names", "n_variants", "cfg_sha16",
                                                    "token_sha16", "traj_sha16")}
            self.KV, self.V = int(b["n_anchors"]), int(b["n_variants"])
            if self.n_var > self.KV * (self.V - 1):
                raise ValueError(f"n_var {self.n_var} > {self.KV} x {self.V - 1} non-identity variants")
            for nm in ("traj", "labels", "ok"):
                self._paths[f"var_{nm}"] = vd / f"{nm}.npy"
            vs = np.load(self._paths["var_labels"], mmap_mode="r").shape
            if vs != (self.n, self.KV * self.V, len(CM.LABEL_COLS)):
                raise ValueError(f"{vd}/labels.npy {vs}: expected ({self.n}, {self.KV * self.V}, 9)")
            ix = np.load(vd / "index.npz")
            self._var_anchor = np.asarray(ix["anchor_idx"], np.int64)          # [N, KV]
            self._var_valid = np.asarray(ix["valid"], bool)                     # [N, KV, V]
            keep &= np.asarray(ix["built"], bool)
            keep &= np.asarray(np.load(self._paths["var_ok"], mmap_mode="r")).any(1)
        self.n_dropped = int(self.n - keep.sum())
        self.gt = bool(gt)
        self._mm: Dict[str, np.ndarray] = {}
        self._cache, self._gtl, self._pid = None, None, None
        r = np.flatnonzero(keep) if rows is None else np.asarray(rows, np.int64)
        if rows is not None and not keep[r].all():
            raise ValueError(f"{int((~keep[r]).sum())} requested rows lack GT / raw256 labels / variants")
        self.rows = r[: int(limit)] if limit else r

    # ------------------------------------------------------------------------------------------- lazy per process
    def _arr(self, name: str) -> np.ndarray:
        if self._pid != os.getpid():
            self._mm, self._cache, self._gtl, self._pid = {}, None, None, os.getpid()
        a = self._mm.get(name)
        if a is None:
            a = self._mm[name] = np.load(self._paths[name], mmap_mode="r")
        return a

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_mm"], d["_cache"], d["_gtl"], d["_pid"] = {}, None, None, None
        return d

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def describe(self) -> Dict:
        return {"split": self.split, "bev": self.bev, "n_rows": int(len(self.rows)), "n_dropped": self.n_dropped,
                "ep_target": self.ep_target,
                "K": self.K, "n_var": self.n_var, "seed": self.seed, "sampler": self.sampler.config(),
                "raw_labels": str(self._paths["raw_labels"]), "var_dir": str(self.var_dir) if self.var_dir else None,
                "var_meta": self.var_meta}

    def _bev_cache(self):
        if self._cache is None:
            if self.bev == "T":
                from navsim.agents.para_ssr.refiner.data import TeacherCache
                self._cache = TeacherCache.for_subset(self.sp["teacher_subset"])
            elif self.bev == "M":
                from navsim.agents.para_ssr.refiner.resmap_cache import ResmapCache
                self._cache = ResmapCache.for_subset(self.sp["teacher_subset"])
        return self._cache

    def load_bev(self, tok: str) -> Tuple[np.ndarray, bool]:
        self._arr("status")                                   # resets per-process state if forked
        try:
            x = np.ascontiguousarray(self._bev_cache().load_bev(tok, s_grid=True), dtype=np.float16)
            if x.shape != BEV_SHAPE:
                raise ValueError(f"{tok}: bev {x.shape}")
            return x, True
        except (FileNotFoundError, KeyError, ValueError, OSError):
            return np.zeros(BEV_SHAPE, np.float16), False

    def __len__(self) -> int:
        return len(self.rows)

    # ------------------------------------------------------------------------------------------- items
    def raw_labels(self, r: int) -> np.ndarray:
        """[256, 9] official labels of the raw anchors of packed row r (LABEL_COLS)."""
        return np.array(self._arr("raw_labels")[r], np.float32)

    def sample_anchors(self, r: int, epoch: Optional[int] = None, lab256: Optional[np.ndarray] = None) -> Dict:
        """AnchorSampler draw on the OFFICIAL labels (pass = NC == DAC == TTC == C == 1, EP unused), any ep_target."""
        lab = self.raw_labels(r) if lab256 is None else lab256
        gt = np.asarray(self._arr("gt_traj")[r], np.float32)
        e = self.epoch if epoch is None else int(epoch)
        return self.sampler.sample(str(self.tokens[r]), gt, lab[:, list(CK_LABEL_IDX)].T, epoch=e, shuffle=True)

    def variant_cols(self, r: int, epoch: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
        """-> (cols [n_var] into the variant files, ok_pad bool [n_var] (False = padding), n_valid)."""
        valid = self._var_valid[r].copy()                     # [KV, V]
        valid[:, 0] = False                                   # identity = the anchor itself (in the K anchors)
        cols = np.flatnonzero(valid.reshape(-1))              # k-major (GT-rank) order: permuted below
        e = self.epoch if epoch is None else int(epoch)
        rng = AS.token_rng(str(self.tokens[r]), e, self.seed, VAR_STREAM)
        cols = cols[rng.permutation(len(cols))]
        nv = len(cols)
        if nv >= self.n_var:
            return cols[: self.n_var], np.ones(self.n_var, bool), nv
        if nv == 0:
            return np.zeros(self.n_var, np.int64), np.zeros(self.n_var, bool), 0
        pad = np.resize(cols, self.n_var)
        return pad, np.arange(self.n_var) < nv, nv

    def labels_item(self, i: int, epoch: Optional[int] = None) -> Dict:
        """Everything except BEV / surrogate GT (cheap; used by __getitem__ and the label prior)."""
        r = int(self.rows[i])
        A = self._arr
        lab = self.raw_labels(r)
        s = self.sample_anchors(r, epoch, lab)
        idx = s["idx"]
        L = lab[idx]
        y = np.ascontiguousarray(ck_targets(L, self.ep_target))
        it = dict(token=str(self.tokens[r]), row=r,
                  cand=np.ascontiguousarray(self.sampler.anchors[idx], np.float32),
                  status=np.array(A("status")[r], np.float32), gt_traj=np.array(A("gt_traj")[r], np.float32),
                  y=y, y_pdms=L[:, PDMS_IDX].copy(),
                  y_ok=np.asarray(A("raw_ok")[r, idx], bool) & np.isfinite(y).all(1),
                  anc_idx=idx.astype(np.int64), anc_group=s["group"].astype(np.int8),
                  anc_pi=s["pi"].astype(np.float64), anc_w=s["w"].astype(np.float32))
        if self.n_var > 0:
            cols, okp, nv = self.variant_cols(r, epoch)
            srt = np.argsort(cols, kind="stable")             # sorted mmap read, then back to the random order
            inv = np.empty_like(srt)
            inv[srt] = np.arange(len(srt))
            vt = np.asarray(A("var_traj")[r, cols[srt]], np.float32)[inv]
            vl = np.asarray(A("var_labels")[r, cols[srt]], np.float32)[inv]
            vo = np.asarray(A("var_ok")[r, cols[srt]], bool)[inv]
            k, v = cols // self.V, cols % self.V
            vy = np.ascontiguousarray(ck_targets(vl, self.ep_target))
            it.update(var_traj=np.ascontiguousarray(vt), var_y=vy,
                      var_pdms=vl[:, PDMS_IDX].copy(),
                      var_ok=vo & okp & np.isfinite(vy).all(1),
                      var_anchor=self._var_anchor[r, k].astype(np.int64), var_v=v.astype(np.int8),
                      var_col=cols.astype(np.int64),
                      var_pi=np.full(self.n_var, min(1.0, self.n_var / nv) if nv else 0.0, np.float32))
        return it

    def __getitem__(self, i: int) -> Dict:
        it = self.labels_item(i)
        if self.bev != "none":
            it["bev"], it["bev_ok"] = self.load_bev(it["token"])
        if self.gt:
            if self._gtl is None or self._pid != os.getpid():
                from navsim.agents.para_ssr.refiner.e2e import GTLoader
                self._arr("status")
                self._gtl = GTLoader(CM.DATA_ROOT)
            it.update(self._gtl.load(it["token"]))
        return it


def ck2_label_prior(ds: CK2Dataset, n_max: int = 4000) -> Optional[np.ndarray]:
    """Expected unweighted training label mean per CK key: epoch-0 sampled anchors + (n_var / n_valid) x every valid
    non-identity variant, over <= n_max evenly spaced rows (score-head bias init = logit(prior)); targets per
    ds.ep_target (the EP prior is the mean of the EP target the head is trained on)."""
    n = len(ds)
    if n == 0:
        return None
    ii = np.unique(np.linspace(0, n - 1, min(n_max, n)).astype(np.int64))
    num, den = np.zeros(5), 0.0
    for i in ii:
        it = ds.labels_item(int(i), epoch=0)
        m = it["y_ok"]
        num += it["y"][m].sum(0)
        den += m.sum()
        if ds.n_var > 0:
            r = int(ds.rows[i])
            valid = ds._var_valid[r].copy()
            valid[:, 0] = False
            cols = np.flatnonzero(valid.reshape(-1))
            if len(cols):
                vl = ck_targets(np.asarray(ds._arr("var_labels")[r, cols], np.float64), ds.ep_target)
                vo = np.asarray(ds._arr("var_ok")[r, cols], bool) & np.isfinite(vl).all(1)
                f = min(1.0, ds.n_var / len(cols))
                num += f * vl[vo].sum(0)
                den += f * vo.sum()
    if den <= 0:
        return None
    return np.clip(num / den, 1e-3, 1 - 1e-3)
