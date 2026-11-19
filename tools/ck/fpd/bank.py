"""FPD teacher bank: the layout of run_dir/bank/<split>/ and the helpers / datasets shared by train_fpd_teacher.py (stages
bank and scorer) and label_fpd_bank.py.

run_dir/bank/<split>/   (row = packed/<split>/tokens.parquet row; N = packed rows of the split)
  sampling (train_fpd_teacher.py --stage bank; GPU, one writer per row)
    tokens.parquet     token, log of every packed row (what the labeler needs for the metric-cache path)
    traj.npy           f32 [N, 32, 8, 3]   column c = 2 k + d: DiT draw d (0, 1) of parent k (NaN = not sampled)
    parent_idx.npy     int64 [N, 16]       anchor ids of the 16 parents = the near + mid anchors of the AnchorSampler's
                                           epoch-0 draw in GT-rank order (the label_variants anchor_idx order); -1 = not
                                           sampled
    sampled.npy        uint8 [N]           0 to do, 1 sampled, 2 BEV unreadable (traj stays NaN, never labelled; a
                                           rerun tries the row again; sample_done.json is refused while more than
                                           train_ck.BEV_BAD_MAX = 1 % of the split's rows are unreadable)
    sample_meta.json   the sampling plan (DiT checkpoint sha16, seed, noise stream, DDIM steps, rows); a resume over a
                       different plan is refused
    sample_done.json   written last (rank 0): counts + traj_sha16 (sha256[:16] of traj.npy)
  labels (label_fpd_bank.py; CPU)
    labels.npy         f32 [N, 32, 9]  LABEL_COLS (NaN = not scored)
    ok.npy             bool [N, 32]    SCORE_COLS finite (the label_cands / labeler rule)
    label_state.npy    uint8 [N]       0 to do, 1 labelled, 2 error (errors.jsonl; label_fpd_bank.py --retry-errors)
    label_plan.json    the traj_sha16 the labels belong to (a re-sampled bank is refused unless --fresh)
    meta.json          written once no sampled row is left to do: counts, scorer source sha256, traj_sha16

Parents.  The stage-dit / stage-scorer forward takes the 16 near + mid slots of the item (anc_group <= 1) in slot order
(near_mid); the bank and the evaluations take the same 16 anchors in GT-rank order (canonical_near_mid: sorted by the
sampler's GT distance, ties to the lower anchor id = argsort(dist, stable) restricted to the 16), so the bank noise
token_noise(token, (16, 2, 8, 3)) is attached to a parent independently of the slot shuffle.
No GPU / scorer imports here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

N_PARENTS = 16                      # near 8 + mid 8 of the AnchorSampler draw (anc_group <= 1)
BANK_DRAWS = 2                      # DiT draws per parent in the bank
BANK_K = N_PARENTS * BANK_DRAWS     # 32 bank trajectories per token
BANK_STREAM = "fpd.teacher_bank"    # fpd_net.token_noise stream of the bank (and of the evaluation draws)
BANK_NOISE_EPOCH = -1               # token_noise epoch -1 = one fixed noise per token
GROUP_NEAR_MID_MAX = 1              # anchor_sampler GROUP_NEAR 0 / GROUP_MID 1 / GROUP_STRAT 2
N_LABEL_COLS = 9

F_TOKENS = "tokens.parquet"
F_TRAJ = "traj.npy"
F_PARENT = "parent_idx.npy"
F_SAMPLED = "sampled.npy"
F_SAMPLE_META = "sample_meta.json"
F_SAMPLE_DONE = "sample_done.json"
F_LABELS = "labels.npy"
F_OK = "ok.npy"
F_LSTATE = "label_state.npy"
F_LPLAN = "label_plan.json"
F_LMETA = "meta.json"
F_ERRORS = "errors.jsonl"
ERR_FRAC_MAX = 0.01                 # stage scorer refuses a bank with more label errors than this fraction


def bank_dir(run_dir, split: str) -> Path:
    return Path(run_dir) / "bank" / split


def read_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.is_file() else default


def sha16_file(path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()[:16]


def sha16_rows(rows) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(rows, np.int64)).tobytes()).hexdigest()[:16]


def token_table(base):
    """DataFrame (token, log) of every packed row of the base dataset's split (row aligned): CK2Dataset ->
    <root>/packed/<split>/tokens.parquet; a test dataset -> its tokens / logs attributes (log '__fake__' if none)."""
    import pandas as pd
    pk = getattr(base, "pk", None)
    if pk is not None and (Path(pk) / "tokens.parquet").is_file():
        df = pd.read_parquet(Path(pk) / "tokens.parquet", columns=["token", "log"])
    else:
        toks = [str(t) for t in base.tokens]
        logs = getattr(base, "logs", None)
        df = pd.DataFrame({"token": toks, "log": [str(x) for x in logs] if logs is not None else ["__fake__"] * len(toks)})
    df["token"] = df.token.astype(str)
    df["log"] = df.log.astype(str)
    if len(df) != int(base.n):
        raise ValueError(f"token table has {len(df)} rows, the dataset {int(base.n)} packed rows")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- parents
def near_mid(cand: torch.Tensor, anc_group: torch.Tensor) -> torch.Tensor:
    """cand [B, K, P, 3], anc_group [B, K] -> the near + mid anchors [B, 16, P, 3] of every item, slot order kept."""
    m = anc_group.to(cand.device).long() <= GROUP_NEAR_MID_MAX
    n = m.sum(1)
    if not bool((n == N_PARENTS).all()):
        raise ValueError(f"every item needs exactly {N_PARENTS} near + mid anchors (anc_group <= 1), got "
                         f"{n.tolist()}")
    return cand[m].reshape(cand.shape[0], N_PARENTS, *cand.shape[2:])


def canonical_near_mid(item: Dict) -> np.ndarray:
    """slot indices [16] of the item's near + mid anchors in GT-rank order: sorted by the AnchorSampler's GT distance
    (anchor_sampler.gt_distance, float64), ties to the lower anchor id (= the sampler's argsort(dist, stable)), i.e.
    AnchorSampler.sample(..., shuffle=False)['idx'][:16] of the same draw."""
    from navsim.agents.para_ssr.ck.anchor_sampler import gt_distance
    g = np.asarray(item["anc_group"])
    m = np.flatnonzero(g <= GROUP_NEAR_MID_MAX)
    if len(m) != N_PARENTS:
        raise ValueError(f"{item.get('token')}: {len(m)} near + mid anchors, expected {N_PARENTS}")
    d = gt_distance(np.asarray(item["cand"])[m], np.asarray(item["gt_traj"]))
    return m[np.lexsort((np.asarray(item["anc_idx"])[m], d))]


def add_parents(item: Dict) -> Dict:
    o = canonical_near_mid(item)
    item["parents"] = np.ascontiguousarray(np.asarray(item["cand"], np.float32)[o])
    item["parent_idx"] = np.asarray(item["anc_idx"], np.int64)[o]
    return item


class ParentItems(torch.utils.data.Dataset):
    """base item (CK2Dataset, n_var 0) reduced to the bank inputs: token, row, bev, bev_ok, status, gt_traj, parents
    [16, 8, 3] (canonical_near_mid), parent_idx [16]."""

    KEYS = ("bev", "bev_ok", "status", "gt_traj")

    def __init__(self, base):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> Dict:
        it = add_parents(self.base[i])
        out = {"token": it["token"], "row": it["row"], "parents": it["parents"], "parent_idx": it["parent_idx"]}
        out.update({k: it[k] for k in self.KEYS})
        return out


# ----------------------------------------------------------------------------------------------- bank dataset
class BankDataset(torch.utils.data.Dataset):
    """base item (CK2Dataset, n_var 0: 32 raw anchors cand / y / y_ok, ...) + the token's bank:
    bank_traj f32 [32, 8, 3], bank_y f32 [32, 5] (ep_target.ck_targets 'official' of labels.npy), bank_ok bool [32]
    (ok.npy & finite targets), bank_pdms f32 [32] (official).  Rows whose bank is not labelled (label_state != 1) or
    not finite are left out (n_dropped): the scorer has no candidate mask, a NaN candidate would reach every other
    candidate of the set through the self-attention.  parents=True also adds parents / parent_idx (add_parents)."""

    def __init__(self, base, bdir, parents: bool = False):
        from navsim.agents.para_ssr.ck.constants import LABEL_COLS
        self.base, self.bdir, self.parents = base, Path(bdir), bool(parents)
        self._paths = {"traj": self.bdir / F_TRAJ, "labels": self.bdir / F_LABELS, "ok": self.bdir / F_OK,
                       "state": self.bdir / F_LSTATE}
        for p in self._paths.values():
            if not p.is_file():
                raise FileNotFoundError(f"bank file {p} missing (stage bank + label_fpd_bank.py first)")
        self.pdms_col = LABEL_COLS.index("pdms")
        rows = np.asarray(base.rows, np.int64)
        st = np.load(self._paths["state"], mmap_mode="r")
        tr = np.load(self._paths["traj"], mmap_mode="r")
        lab = np.load(self._paths["labels"], mmap_mode="r")
        n = int(base.n)
        if st.shape != (n,) or tr.shape != (n, BANK_K, 8, 3) or lab.shape != (n, BANK_K, N_LABEL_COLS):
            raise ValueError(f"{self.bdir}: bank shapes {st.shape} {tr.shape} {lab.shape} do not match {n} packed rows")
        good = np.zeros(len(rows), bool)
        for lo in range(0, len(rows), 8192):                  # chunked: traj of the train split is 261 MB
            r = rows[lo:lo + 8192]
            good[lo:lo + 8192] = (np.asarray(st[r]) == 1) & np.isfinite(np.asarray(tr[r])).reshape(len(r), -1).all(1)
        self.pos = np.flatnonzero(good)
        self.rows = rows[self.pos]
        self.n = n
        self.n_dropped = int((~good).sum())
        self._mm: Dict[str, np.ndarray] = {}
        self._pid = None

    def _arr(self, name: str) -> np.ndarray:
        if self._pid != os.getpid():
            self._mm, self._pid = {}, os.getpid()
        a = self._mm.get(name)
        if a is None:
            a = self._mm[name] = np.load(self._paths[name], mmap_mode="r")
        return a

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_mm"], d["_pid"] = {}, None
        return d

    def __len__(self) -> int:
        return len(self.pos)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    @property
    def epoch(self) -> int:
        return int(getattr(self.base, "epoch", 0))

    def describe(self) -> Dict:
        b = self.base.describe() if hasattr(self.base, "describe") else {}
        return {"base": b, "bank_dir": str(self.bdir), "n_rows": int(len(self.pos)), "n_dropped_bank": self.n_dropped,
                "bank_k": BANK_K}

    def bank(self, r: int) -> Dict:
        from navsim.agents.para_ssr.ck.ep_target import ck_targets
        tr = np.array(self._arr("traj")[r], np.float32)
        lab = np.array(self._arr("labels")[r], np.float32)
        y = np.ascontiguousarray(ck_targets(lab, "official"))
        ok = np.asarray(self._arr("ok")[r], bool) & np.isfinite(y).all(1)
        return {"bank_traj": tr, "bank_y": y, "bank_ok": ok, "bank_pdms": lab[:, self.pdms_col].copy()}

    def labels_item(self, j: int, epoch: Optional[int] = None) -> Dict:
        it = self.base.labels_item(int(self.pos[j]), epoch=epoch)
        it.update(self.bank(int(it["row"])))
        return it

    def __getitem__(self, j: int) -> Dict:
        it = self.base[int(self.pos[j])]
        it.update(self.bank(int(it["row"])))
        return add_parents(it) if self.parents else it


def bank_label_prior(ds: BankDataset, n_max: int = 4000) -> Optional[np.ndarray]:
    """Expected training label mean per CK key over <= n_max evenly spaced items: the 32 raw anchors of the sampler's
    epoch-0 draw + the 32 bank trajectories (ok candidates only, official targets) -> score-head bias logit(prior)."""
    n = len(ds)
    if n == 0:
        return None
    ii = np.unique(np.linspace(0, n - 1, min(int(n_max), n)).astype(np.int64))
    num, den = np.zeros(5), 0.0
    for i in ii:
        it = ds.labels_item(int(i), epoch=0)
        for y, ok in ((it["y"], it["y_ok"]), (it["bank_y"], it["bank_ok"])):
            m = np.asarray(ok, bool) & np.isfinite(y).all(1)
            num += np.asarray(y, np.float64)[m].sum(0)
            den += float(m.sum())
    if den <= 0:
        return None
    return np.clip(num / den, 1e-3, 1 - 1e-3)


# ----------------------------------------------------------------------------------------------- checks
def bank_status(bdir) -> Dict:
    """What is on disk for one split: sample / label meta (None when absent)."""
    bdir = Path(bdir)
    return {"dir": str(bdir), "sample_meta": read_json(bdir / F_SAMPLE_META), "sample_done": read_json(bdir / F_SAMPLE_DONE),
            "label_plan": read_json(bdir / F_LPLAN), "label_meta": read_json(bdir / F_LMETA)}


def check_bank(bdir, dit_sha16: Optional[str] = None, err_frac_max: float = ERR_FRAC_MAX,
               rows_sha16: Optional[str] = None) -> Dict:
    """Stage scorer prerequisites of one split: sampling finished (sample_done.json), labelling finished (meta.json),
    the labels belong to the current traj.npy (traj_sha16), the bank was sampled from the given DiT checkpoint for the
    given dataset rows (sha16_rows of the scorer's dataset: another --limit-tokens / --val-limit would silently train
    on the sampled subset only), and at most err_frac_max of the sampled rows failed to label.  -> summary dict;
    SystemExit otherwise."""
    s = bank_status(bdir)
    done, meta = s["sample_done"], s["label_meta"]
    if done is None:
        raise SystemExit(f"{bdir}: sampling not finished (no {F_SAMPLE_DONE}): run --stage bank")
    if meta is None:
        raise SystemExit(f"{bdir}: labels not finished (no {F_LMETA}): run tools/ck/fpd/label_fpd_bank.py")
    if meta.get("traj_sha16") != done.get("traj_sha16"):
        raise SystemExit(f"{bdir}: labels are for traj sha16 {meta.get('traj_sha16')} but traj.npy is "
                         f"{done.get('traj_sha16')} (re-sampled bank): relabel with label_fpd_bank.py --fresh")
    if dit_sha16 is not None and done.get("dit_ckpt_sha16") != dit_sha16:
        raise SystemExit(f"{bdir}: sampled from DiT checkpoint {done.get('dit_ckpt_sha16')} but dit/ckpt_last.pt is "
                         f"{dit_sha16}: move bank/{Path(bdir).name} away and rerun --stage bank")
    if rows_sha16 is not None and done.get("rows_sha16") != rows_sha16:
        raise SystemExit(f"{bdir}: sampled for dataset rows sha16 {done.get('rows_sha16')} ({done.get('n_rows')} rows) "
                         f"but this run's rows are {rows_sha16} (another --limit-tokens / --val-limit or data): move "
                         f"bank/{Path(bdir).name} away and rerun --stage bank")
    n_s = int(meta.get("n_sampled", 0))
    n_e = int(meta.get("n_err", 0))
    if n_s <= 0:
        raise SystemExit(f"{bdir}: no labelled row")
    if n_e > err_frac_max * n_s:
        raise SystemExit(f"{bdir}: {n_e} of {n_s} rows failed to label (> {100 * err_frac_max:g} %): "
                         f"label_fpd_bank.py --retry-errors (see {F_ERRORS})")
    return {"dir": str(bdir), "traj_sha16": done.get("traj_sha16"), "dit_ckpt_sha16": done.get("dit_ckpt_sha16"),
            "n_rows": done.get("n_rows"), "n_sampled": n_s, "n_labelled": meta.get("n_labelled"), "n_err": n_e,
            "n_bev_bad": done.get("n_bev_bad"), "n_ok_cands": meta.get("n_ok_cands")}


def stage_rows_limit(split: str, train_split: str, val_split: str, limit_tokens: int, val_limit: int) -> int:
    """The dataset limit of a split (first N rows): --limit-tokens for the training split, --val-limit for the
    validation split, 0 (all rows) otherwise."""
    if split == train_split:
        return int(limit_tokens or 0)
    if split == val_split:
        return int(val_limit or 0)
    return 0


def parse_list(s) -> List[str]:
    if isinstance(s, (list, tuple)):
        return [str(x) for x in s]
    return [x.strip() for x in str(s or "").split(",") if x.strip()]
