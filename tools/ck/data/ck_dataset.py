"""CK Phase 1 dataset: one item = one token with its K v2 candidates (+ BEV, labels, GT, KD targets, lead flags).

CKDataset(split, bev in {'S','T','M','none'}, k=16, labels='cand', gt=False, kd_dir=None, corr_aug=False, lead=False,
          rows=None, limit=0, root=None, ep_target='official')
  rows      packed row indices (default: every packed row with ok); limit keeps the first `limit` of them
  bev       S: v2 dump <root>/v2dump/<split>/bev/<tok[:2]>/<tok>.npy (f16 S grid)
            T: refiner.data.TeacherCache.for_subset(teacher_subset).load_bev(tok, s_grid=True)   (BEVFusion t0)
            M: refiner.resmap_cache.ResmapCache.for_subset(teacher_subset).load_bev(tok, s_grid=True) (ReSMap;
               raises for navtrain_val: no cache)
            none: no BEV key.  A token whose BEV cannot be read gets zeros and bev_ok False.
  labels    <root>/labels/<split>/<labels>/ (labels.npy [N,K,9], ok.npy [N,K]); None -> no y keys
  ep_target 'official' (default, old behaviour bit-identical) | 'decoupled': y / y_corr = ck.ep_target.ck_targets(
            labels, ep_target) (EP without the NC * DAC * DDC factor); with 'decoupled' y_ok also requires finite y.
            y_pdms stays official.
  gt        refiner.e2e.GTLoader(DATA_ROOT).load(token) keys (ref_*; fixed padded shapes, ref_gt_ok)
  kd_dir    kd_targets/<tag> dir (absolute or relative to root): kd_score_prob [K,5], kd_c_lon [K,6], kd_e_lat [K,6],
            kd_ok [K]; its tokens.parquet must equal the packed order
  corr_aug  corr_traj [K,8,3] (kd_dir/kd_corr_traj.npy), y_corr [K,5] / y_corr_ok [K] (labels/<split>/kd_corr),
            kd_score_prob_corr [K,5] (kd_dir)
  lead      lead_has, lead_d1 (f32; NaN = censored) from <root>/lead/<split>.parquet (fallback CK_DATA/lead)
  objects   refiner-v3 temporal branch (default None = the CK2 path, no new keys): {"cfg": TemporalCfg dict (on),
            "aug": False, "targets": bool} -> obj_in / obj_mask / frame_valid / obj_ok (+ fc_xyh / fc_state, + the
            oracle's priv_fc_*) from the object pack (ck2_dataset.ObjectRows).  Evaluation only: aug must be False
            (CKDataset has no epoch / seed; training augmentation lives in CK2Dataset).
Item keys: token (str), row (int), bev f16 [256,50,100] + bev_ok, cand f32 [K,8,3], status f32 [8], v2_final f32 [K],
  v2_im f32 [K], v2_sim f32 [K,5] (CK_KEYS), gt_traj f32 [8,3] (always; NaN if unavailable), y f32 [K,5] (CK_KEYS),
  y_ok bool [K], y_pdms f32 [K], + the optional groups above.
collate_ck(items) -> {'tokens': [str], 'rows': int64 [T], every other key stacked with a leading T dim (torch)}.
make_loader(ds, tokens_per_batch, shuffle, workers, seed) -> DataLoader (persistent workers, pin_memory, drop_last on
  train, seeded generator; reseed per epoch with loader.generator.manual_seed(seed + epoch)).
Arrays are memory-mapped lazily per process (safe with fork or spawn workers).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root of this worktree
from tools.ck.data import common as CM  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from navsim.agents.para_ssr.ck.ep_target import check_ep_target, ck_targets  # noqa: E402

BEV_SHAPE = (256, 50, 100)
CK_LABEL_IDX = tuple(CM.C.CK_LABEL_IDX)            # labels[..., CK_LABEL_IDX] -> CK_KEYS order
PDMS_IDX = CM.LABEL_COLS.index("pdms")


class CKDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, bev: str = "S", k: int = CM.K_CAND, labels: Optional[str] = "cand",
                 gt: bool = False, kd_dir: Optional[str] = None, corr_aug: bool = False, lead: bool = False,
                 rows: Optional[np.ndarray] = None, limit: int = 0, root=None, ep_target: str = "official",
                 objects: Optional[Dict] = None):
        assert bev in ("S", "T", "M", "none"), bev
        self.split, self.bev, self.k = split, bev, int(k)
        self.ep_target = check_ep_target(ep_target)
        self.sp = CM.split(split)
        self.root = CM.root_dir(root)
        self.pk = CM.packed_dir(split, self.root)
        if not (self.pk / "tokens.parquet").is_file():
            raise FileNotFoundError(f"{self.pk}/tokens.parquet missing (run pack_v2.py --split {split})")
        tdf = pd.read_parquet(self.pk / "tokens.parquet")
        self.tokens = tdf.token.astype(str).to_numpy()
        self.n = len(self.tokens)
        self._paths: Dict[str, Path] = {n: self.pk / f"{n}.npy" for n in
                                        ("cand", "v2_final", "v2_im", "v2_sim", "status", "gt_traj", "ok")}
        kp = np.load(self._paths["cand"], mmap_mode="r").shape
        assert kp[0] == self.n and kp[1] >= self.k, (kp, self.n, self.k)
        ok = np.load(self._paths["ok"])
        if bev == "M" and not self.sp.get("map_teacher", False):
            raise ValueError(f"no ReSMap teacher cache for {split} (map_teacher False)")
        self.labels = labels
        if labels:
            ld = CM.labels_dir(split, labels, self.root)
            self._paths["labels"], self._paths["labels_ok"] = ld / "labels.npy", ld / "ok.npy"
            ls = np.load(self._paths["labels"], mmap_mode="r").shape
            assert ls[0] == self.n and ls[1] >= self.k, (ls, self.n)
        self.kd_dir = None
        if kd_dir:
            kd = Path(kd_dir)
            kd = kd if kd.is_absolute() else self.root / kd
            self.kd_dir = kd
            kt = pd.read_parquet(kd / "tokens.parquet").token.astype(str).to_numpy()
            assert len(kt) == self.n and (kt == self.tokens).all(), f"{kd}: token order differs from packed {split}"
            for n in ("kd_score_prob", "kd_c_lon", "kd_e_lat", "kd_ok"):
                self._paths[n] = kd / f"{n}.npy"
        self.corr_aug = bool(corr_aug)
        if self.corr_aug:
            assert self.kd_dir is not None, "corr_aug needs kd_dir"
            self._paths["corr_traj"] = self.kd_dir / "kd_corr_traj.npy"
            self._paths["kd_score_prob_corr"] = self.kd_dir / "kd_score_prob_corr.npy"
            lc = CM.labels_dir(split, "kd_corr", self.root)
            self._paths["y_corr"], self._paths["y_corr_ok"] = lc / "labels.npy", lc / "ok.npy"
            for n in ("corr_traj", "kd_score_prob_corr", "y_corr", "y_corr_ok"):
                assert self._paths[n].is_file(), f"corr_aug: {self._paths[n]} missing"
        self.lead = bool(lead)
        if self.lead:
            lp = CM.lead_path(split, self.root)
            if not lp.is_file():            # lead flags are per split, not per root: smoke roots reuse CK_DATA/lead
                lp = CM.lead_path(split, CM.CK_DATA)
            ldf = pd.read_parquet(lp).set_index("token")
            m = ldf.reindex(self.tokens)
            self._lead_has = m.has_lead.to_numpy(np.float32)
            self._lead_d1 = m.D_1.to_numpy(np.float32)
        self.gt = bool(gt)
        self._gtl = None
        self._cache = None
        self._mm: Dict[str, np.ndarray] = {}
        self._pid = None
        r = np.flatnonzero(ok) if rows is None else np.asarray(rows, np.int64)
        self.rows = r[: int(limit)] if limit else r
        self.objects = None
        if objects is not None:
            from tools.ck.data.ck2_dataset import ObjectRows
            if bool(objects.get("aug", False)) if isinstance(objects, dict) else False:
                raise ValueError("CKDataset objects: aug must be False (evaluation dataset; train with CK2Dataset)")
            self.objects = ObjectRows(split, objects, self.tokens, self.rows)

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

    def _bev_cache(self):
        if self._cache is None:
            if self.bev == "T":
                from navsim.agents.para_ssr.refiner.data import TeacherCache
                self._cache = TeacherCache.for_subset(self.sp["teacher_subset"])
            elif self.bev == "M":
                from navsim.agents.para_ssr.refiner.resmap_cache import ResmapCache
                self._cache = ResmapCache.for_subset(self.sp["teacher_subset"])
        return self._cache

    def load_bev(self, tok: str):
        try:
            if self.bev == "S":
                x = np.load(CM.bev_path(CM.dump_dir(self.split, self.root), tok), mmap_mode="r")
            else:
                x = self._bev_cache().load_bev(tok, s_grid=True)
            x = np.ascontiguousarray(x, dtype=np.float16)
            if x.shape != BEV_SHAPE:
                raise ValueError(f"{tok}: bev {x.shape}")
            return x, True
        except (FileNotFoundError, KeyError, ValueError, OSError):
            return np.zeros(BEV_SHAPE, np.float16), False

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict:
        r = int(self.rows[i])
        k = self.k
        tok = str(self.tokens[r])
        A = self._arr
        it = dict(token=tok, row=r,
                  cand=np.array(A("cand")[r, :k], np.float32), status=np.array(A("status")[r], np.float32),
                  v2_final=np.array(A("v2_final")[r, :k], np.float32), v2_im=np.array(A("v2_im")[r, :k], np.float32),
                  v2_sim=np.array(A("v2_sim")[r, :k], np.float32), gt_traj=np.array(A("gt_traj")[r], np.float32))
        if self.bev != "none":
            it["bev"], it["bev_ok"] = self.load_bev(tok)
        if self.labels:
            lab = np.array(A("labels")[r, :k], np.float32)
            it["y"] = ck_targets(lab, self.ep_target)
            it["y_ok"] = np.array(A("labels_ok")[r, :k], bool)
            if self.ep_target != "official":
                it["y_ok"] &= np.isfinite(it["y"]).all(-1)
            it["y_pdms"] = lab[:, PDMS_IDX]
        if self.kd_dir is not None:
            it["kd_score_prob"] = np.array(A("kd_score_prob")[r, :k], np.float32)
            it["kd_c_lon"] = np.array(A("kd_c_lon")[r, :k], np.float32)
            it["kd_e_lat"] = np.array(A("kd_e_lat")[r, :k], np.float32)
            it["kd_ok"] = np.array(A("kd_ok")[r, :k], bool)
        if self.corr_aug:
            it["corr_traj"] = np.array(A("corr_traj")[r, :k], np.float32)
            it["y_corr"] = ck_targets(np.array(A("y_corr")[r, :k], np.float32), self.ep_target)
            it["y_corr_ok"] = np.array(A("y_corr_ok")[r, :k], bool)
            it["kd_score_prob_corr"] = np.array(A("kd_score_prob_corr")[r, :k], np.float32)
        if self.lead:
            it["lead_has"] = np.float32(self._lead_has[r])
            it["lead_d1"] = np.float32(self._lead_d1[r])
        if self.gt:
            if self._gtl is None or self._pid != os.getpid():
                from navsim.agents.para_ssr.refiner.e2e import GTLoader
                self._arr("cand")       # resets per-process state if forked
                self._gtl = GTLoader(CM.DATA_ROOT)
            it.update(self._gtl.load(tok))
        if self.objects is not None:
            it.update(self.objects.item(r, tok, 0, 0))
        return it


def collate_ck(items: List[Dict]) -> Dict:
    out = {"tokens": [it["token"] for it in items],
           "rows": torch.as_tensor([it["row"] for it in items], dtype=torch.int64)}
    for key in items[0]:
        if key in ("token", "row"):
            continue
        v0 = items[0][key]
        if isinstance(v0, torch.Tensor):
            out[key] = torch.stack([it[key] for it in items])
        else:
            out[key] = torch.as_tensor(np.stack([np.asarray(it[key]) for it in items]))
    return out


def make_loader(ds: CKDataset, tokens_per_batch: int = 8, shuffle: bool = True, workers: int = 8, seed: int = 0,
                drop_last: Optional[bool] = None) -> torch.utils.data.DataLoader:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        ds, batch_size=int(tokens_per_batch), shuffle=shuffle, collate_fn=collate_ck, num_workers=int(workers),
        pin_memory=True, drop_last=shuffle if drop_last is None else drop_last, generator=g,
        persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None)
