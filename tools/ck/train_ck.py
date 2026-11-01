#!/usr/bin/env python
"""Train one CK net (report 44 §4; contract pipeline.train_ck): teacher DET (arm T, BEVFusion t0), teacher MAP (arm M,
ReSMap) or the student (arm S, frozen v2 r34 bev_embed dump) on navtrain train logs.

  cd /workspace/yongjae/SSR-ck2 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/yongjae/SSR-ck2 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 /venv/ssr/bin/python tools/ck/train_ck.py --arm T --run ckT_p1 --resume

Loss (outside autocast, f32):
  L = lambda_score * score_bce(orig)                                     [GT on-policy labels, 5 keys]
    + [corr_aug]  lambda_corr_score * score_bce(corr)                    [tau'_KD scored officially: labels kd_corr]
    + lambda_sur * correction_loss(decode(orig, z, w, slope 0.1))        [surrogate on GT-ok tokens]
    + [kd on]     lambda_kd_score * (score_kd_bce(orig) + score_kd_bce(corr if corr_aug))
    + [kd on]     w_kd * ctrl_kd_l1(c_lon[2:], e_lat[2:]; lon 0.25, lat 1.0)   (w_kd = EMA balance, E2 rule)
    + [lead_aux]  lambda_lead * BCE(lead_logit, D_1 | has_lead, uncensored)
Teachers (T, M): kd off, no corr_aug.
Optimiser: AdamW, linear warmup --warmup-steps then cosine to 0; fp16 autocast + GradScaler; clip; a step whose loss or
grad norm is non-finite is skipped and counted.
Eval every epoch (and at a --max-steps stop): --val-split (unless none) and a fixed log-stratified subset of
--train-eval-rows train rows -> val_metrics.jsonl ('split' field; train-vs-val gap of the DET teacher).
Outputs CK_DATA/train/<run>/: config.json, norm.npz | norm_map.npz, ckpt_last.pt (every --ckpt-every steps and every
epoch end; mid-epoch resume continues the same seeded batch order), ckpt_ep<e>.pt, train_log.jsonl, val_metrics.jsonl,
done.json.  No val-based checkpoint selection (ckpt_last.pt is the final model).
The epoch-end ckpt_last.pt is written with an 'eval_pending' marker before the evaluations and re-written without it
after them; --resume of a marked checkpoint first runs the evaluations still missing from val_metrics.jsonl.
Unreadable BEVs (CKDataset bev_ok False) are masked out of every loss, counted (n_bev_bad in train_log.jsonl /
val_metrics.jsonl) and stop the run (SystemExit) when they exceed 1 % of the tokens seen.
"""
from __future__ import annotations

import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[2])   # repo root of this worktree
if sys.path[0] != CK:
    sys.path.insert(0, CK)
import navsim  # noqa: E402

assert navsim.__file__.startswith(CK), navsim.__file__

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable, Dict, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from tools.ck import ckutil as U  # noqa: E402

ARMS = ("T", "M", "S")
NORM_FILE = {"T": "norm.npz", "M": "norm_map.npz"}
TEACHER_INIT = {"T": "stageE/teachers/stageT4_T_fold0_seed0", "M": "stageE/teachers/stageT4_M_fold0_seed0"}
TRAIN_EVAL_NAME = "{}_train_eval"


def C():
    from navsim.agents.para_ssr.ck import constants
    return constants


# ----------------------------------------------------------------------------------------------- args
def get_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="train a CK net (teacher T / M or student S)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-root", default=None, help="default CK_DATA/train")
    ap.add_argument("--split", default="navtrain_train")
    ap.add_argument("--val-split", default=None, help="navtrain_val | none (default: T,S navtrain_val; M none)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--tokens-per-batch", type=int, default=8)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kd", choices=["off", "on"], default=None, help="default: S on, T/M off")
    ap.add_argument("--kd-dir", default=None, help="default (S, kd on): CK_DATA/kd_targets/phase1")
    ap.add_argument("--corr-aug", choices=["auto", "on", "off"], default=None, help="default: S auto, T/M off")
    ap.add_argument("--init-from", default=None,
                    help="v1 stage-T run dir or CK run dir; 'none' = scratch (default: T/M stageT4 snapshot, S none)")
    ap.add_argument("--lambda-score", type=float, default=None)
    ap.add_argument("--lambda-corr-score", type=float, default=None)
    ap.add_argument("--lambda-sur", type=float, default=None)
    ap.add_argument("--lambda-kd-score", type=float, default=None)
    ap.add_argument("--kd-ratio", type=float, default=None)
    ap.add_argument("--kd-ctrl-balance", choices=["ema", "fixed"], default=None)
    ap.add_argument("--kd-ctrl-weight", type=float, default=1.0, help="fixed KD control weight (balance 'fixed')")
    ap.add_argument("--lead-aux", type=int, default=0)
    ap.add_argument("--score-prior", type=int, default=1, help="score-head bias = logit(train label mean) unless "
                    "initialised from a CK run")
    ap.add_argument("--lambda-lead", type=float, default=None)
    ap.add_argument("--train-eval-rows", type=int, default=18179)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--limit-tokens", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=2000, help="mid-epoch ckpt_last.pt every N steps (0 = epoch only)")
    ap.add_argument("--device", default="cuda", help="cuda (needs CUDA_VISIBLE_DEVICES in 0-3) | cpu")
    ap.add_argument("--resume", action="store_true", help="continue from ckpt_last.pt if present")
    ap.add_argument("--restart", action="store_true", help="wipe the run dir outputs and start over")
    return ap


def resolve_args(a, kd_exists: Optional[Callable[[str], bool]] = None):
    """Fill arm-dependent defaults in place (contract defaults_by_arm) and return a."""
    Cn = C()
    LD = dict(Cn.LOSS_DEFAULTS)
    data_root, ck_root = Path(Cn.DATA_ROOT), U.ck_data()
    if a.out_root is None:
        a.out_root = str(ck_root / "train")
    if a.val_split is None:
        a.val_split = "none" if a.arm == "M" else "navtrain_val"
    if a.kd is None:
        a.kd = "on" if a.arm == "S" else "off"
    if a.corr_aug is None:
        a.corr_aug = "auto" if a.arm == "S" else "off"
    if a.init_from is None:
        a.init_from = str(data_root / TEACHER_INIT[a.arm]) if a.arm in TEACHER_INIT else "none"
    if a.kd == "on" and a.kd_dir is None:
        a.kd_dir = str(ck_root / "kd_targets" / "phase1")
    for name, key in (("lambda_score", "lambda_score"), ("lambda_corr_score", "lambda_corr_score"),
                      ("lambda_sur", "lambda_sur"), ("lambda_kd_score", "lambda_kd_score"),
                      ("kd_ratio", "kd_ratio"), ("kd_ctrl_balance", "kd_ctrl_balance"), ("lambda_lead", "lambda_lead")):
        if getattr(a, name) is None:
            setattr(a, name, LD[key])
    if a.lead_aux and not a.lambda_lead:
        a.lambda_lead = 1.0
    if a.corr_aug == "auto":
        if kd_exists is None:
            kd_exists = lambda d: (Path(d) / "kd_corr_traj.npy").is_file() and \
                (ck_root / "labels" / a.split / "kd_corr" / "labels.npy").is_file()
        a.corr_aug = "on" if (a.kd_dir and kd_exists(a.kd_dir)) else "off"
    if a.corr_aug == "on" and not a.kd_dir:
        raise SystemExit("--corr-aug on needs --kd-dir (kd_corr_traj.npy + kd_score_prob_corr.npy)")
    if a.arm in ("T", "M") and a.kd == "on":
        raise SystemExit("teachers train without KD (--kd off)")
    return a


# ----------------------------------------------------------------------------------------------- data
class EpochBatches(torch.utils.data.Sampler):
    """Seeded per-epoch batch order (rng(seed * 100003 + epoch)), drop_last, starting at batch `start` (resume)."""

    def __init__(self, n: int, bs: int, seed: int, epoch: int, start: int = 0, shuffle: bool = True,
                 drop_last: bool = True):
        self.n, self.bs, self.seed, self.epoch, self.start = int(n), int(bs), int(seed), int(epoch), int(start)
        self.shuffle, self.drop_last = shuffle, drop_last

    def batches(self) -> List[np.ndarray]:
        order = (np.random.default_rng(self.seed * 100003 + self.epoch).permutation(self.n) if self.shuffle
                 else np.arange(self.n))
        out = [order[i:i + self.bs] for i in range(0, self.n, self.bs)]
        if self.drop_last and out and len(out[-1]) < self.bs:
            out = out[:-1]
        return out

    def __iter__(self):
        for b in self.batches()[self.start:]:
            yield [int(i) for i in b]

    def __len__(self):
        return max(0, len(self.batches()) - self.start)


def default_collate():
    from tools.ck.data.ck_dataset import collate_ck
    return collate_ck


def make_loader(ds, bs: int, seed: int, epoch: int, start: int, shuffle: bool, workers: int, collate_fn,
                pin: bool, drop_last: bool = True):
    from torch.utils.data import DataLoader
    sampler = EpochBatches(len(ds), bs, seed, epoch, start, shuffle, drop_last)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=int(workers), pin_memory=pin,
                      prefetch_factor=4 if workers > 0 else None, persistent_workers=False)


def packed_tokens(split: str):
    import pandas as pd
    p = U.ck_data() / "packed" / split / "tokens.parquet"
    return pd.read_parquet(p) if p.is_file() else None


def build_datasets(a) -> Dict:
    """-> {'train': CKDataset, 'evals': [(name, CKDataset)]} from the data package (contract data.ck_dataset)."""
    from tools.ck.data.ck_dataset import CKDataset

    kd_dir = a.kd_dir if a.kd == "on" or a.corr_aug == "on" else None
    train = CKDataset(a.split, bev=a.arm, k=a.k, labels="cand", gt=True, kd_dir=kd_dir,
                      corr_aug=a.corr_aug == "on", lead=bool(a.lead_aux), limit=a.limit_tokens or 0)
    evals = []
    if a.val_split and a.val_split != "none":
        evals.append((a.val_split, CKDataset(a.val_split, bev=a.arm, k=a.k, labels="cand", gt=False,
                                             limit=a.val_limit or 0)))
    if a.train_eval_rows:
        tdf = packed_tokens(a.split)
        if tdf is not None:
            pool = np.asarray(train.rows, np.int64)          # packed-ok rows actually trained on (after limit)
            n_eval = a.train_eval_rows if not a.val_limit else min(a.train_eval_rows, a.val_limit)
            sub = U.log_stratified_rows(tdf["log"].values[pool], n_eval, seed=0)
            evals.append((TRAIN_EVAL_NAME.format(a.split),
                          CKDataset(a.split, bev=a.arm, k=a.k, labels="cand", gt=False, rows=pool[sub])))
    return {"train": train, "evals": evals}


# ----------------------------------------------------------------------------------------------- model
def load_norm_from(init_from: str, arm: str):
    """(mean, std, path) of the arm's z-score from a v1 / CK run dir, or None."""
    if arm not in NORM_FILE or not init_from or init_from == "none":
        return None
    p = Path(init_from) / NORM_FILE[arm]
    if not p.is_file():
        return None
    from navsim.agents.para_ssr.refiner.adapters import load_norm
    mean, std, _ = load_norm(p)
    return mean, std, p


def compute_norm(arm: str, tokens: Sequence[str], run: Path):
    """Arm T / M z-score over <= 2048 train tokens (seed 0) when no init source provides it; saved to run dir."""
    from navsim.agents.para_ssr.refiner import data as RD
    from navsim.agents.para_ssr.refiner.adapters import save_norm
    if arm == "T":
        cache = RD.TeacherCache.for_subset("navtrain")
    else:
        from navsim.agents.para_ssr.refiner.resmap_cache import ResmapCache
        cache = ResmapCache.for_subset("navtrain")
    mean, std, info = RD.compute_teacher_norm(cache, list(tokens), 2048, seed=0)
    info.update(branch="det" if arm == "T" else "map", source="tools/ck/train_ck.py compute_norm")
    save_norm(run / NORM_FILE[arm], mean, std, info)
    return mean, std


def setup_model(a, run: Path, train_tokens: Optional[Sequence[str]] = None, norm_override=None):
    """-> (CKNet, init_report dict).  Copies the arm's norm file into the run dir (load_ck rebuilds from it)."""
    from navsim.agents.para_ssr.ck.model import build_ck

    norm = None
    if a.arm in NORM_FILE:
        dst = run / NORM_FILE[a.arm]
        if norm_override is not None:
            from navsim.agents.para_ssr.refiner.adapters import save_norm
            if not dst.is_file():
                save_norm(dst, norm_override[0], norm_override[1], {"source": "override (tests)"})
        elif not dst.is_file():
            src = load_norm_from(a.init_from, a.arm)
            if src is not None:
                shutil.copyfile(src[2], dst)
            else:
                if train_tokens is None:
                    raise SystemExit(f"arm {a.arm}: no norm in --init-from and no train tokens to compute it")
                compute_norm(a.arm, train_tokens, run)
        from navsim.agents.para_ssr.refiner.adapters import load_norm
        mean, std, _ = load_norm(dst)
        norm = (mean, std)
    init = None if (not a.init_from or a.init_from == "none") else a.init_from
    net = build_ck(a.arm, seed=a.seed, norm=norm, init_from=init, lead_aux=bool(a.lead_aux))
    rep = getattr(net, "init_report", None)
    rep = dict(rep) if isinstance(rep, dict) else {"report": str(rep)}
    if getattr(a, "score_prior", 1) and rep.get("kind") != "ck":
        prior = label_prior(a.split)
        if prior is not None:
            net.set_score_prior(prior)
            rep["score_prior"] = [float(x) for x in prior]
    return net, rep


def label_prior(split: str, n_max: int = 20000):
    """Mean official label per CK key over ok candidates of <= n_max rows of labels/<split>/cand (score-head bias
    init = logit(prior)); None if the labels are not there."""
    from navsim.agents.para_ssr.ck import constants as Cn
    d = U.ck_data() / "labels" / split / "cand"
    if not (d / "labels.npy").is_file():
        return None
    lab = np.load(d / "labels.npy", mmap_mode="r")
    ok = np.load(d / "ok.npy", mmap_mode="r")
    rows = np.unique(np.linspace(0, len(lab) - 1, min(n_max, len(lab))).astype(np.int64))
    L = np.asarray(lab[rows], np.float64)[..., list(Cn.CK_LABEL_IDX)]
    m = np.asarray(ok[rows], bool) & np.isfinite(L).all(-1)
    if not m.any():
        return None
    return np.clip(L[m].mean(0), 1e-3, 1 - 1e-3)


# ----------------------------------------------------------------------------------------------- loss
def to_dev(batch: Dict, dev) -> Dict:
    return {k: (v.to(dev, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def gt_index(batch: Dict) -> torch.Tensor:
    """tokens with surrogate GT (ref_gt_ok) and a finite GT human trajectory (core losses.surrogate_index)."""
    from navsim.agents.para_ssr.ck.losses import surrogate_index
    if "ref_gt_ok" not in batch or "gt_traj" not in batch:
        return torch.zeros(0, dtype=torch.long, device=batch["cand"].device)
    return surrogate_index(batch["ref_gt_ok"], batch["gt_traj"].float())


BEV_BAD_MAX = 0.01            # > 1 % unreadable BEVs (cache mount / path problem) stops the run (SystemExit)
BEV_BAD_MIN_TOK = 1000        # mid-epoch check once this many tokens were seen; always checked at epoch end


def n_bev_bad(batch: Dict) -> int:
    """number of tokens in the batch whose BEV could not be read (item bev_ok False); 0 without a bev_ok key."""
    if "bev_ok" not in batch:
        return 0
    return int((~batch["bev_ok"].bool().reshape(-1)).sum())


def check_bev_bad(bad: int, tok: int, where: str, force: bool = False) -> None:
    """SystemExit when more than BEV_BAD_MAX of the tokens seen had no BEV (checked once tok >= BEV_BAD_MIN_TOK, or
    always with force)."""
    if tok <= 0 or (tok < BEV_BAD_MIN_TOK and not force):
        return
    if bad / tok > BEV_BAD_MAX:
        raise SystemExit(f"{where}: {bad}/{tok} tokens ({100.0 * bad / tok:.2f} %) had no readable BEV "
                         f"(> {100 * BEV_BAD_MAX:g} %): check the BEV cache / dump paths")


def mask_bev_ok(batch: Dict) -> Dict:
    """Tokens whose BEV could not be loaded (item bev_ok False) drop out of every loss / metric mask."""
    if "bev_ok" not in batch:
        return batch
    b = dict(batch)
    ok = b["bev_ok"].bool().reshape(-1)
    if bool(ok.all()):
        return b
    for k in ("y_ok", "y_corr_ok", "kd_ok"):
        if k in b:
            b[k] = b[k].bool() & ok[:, None]
    if "ref_gt_ok" in b:
        b["ref_gt_ok"] = b["ref_gt_ok"].bool() & ok
    return b


def forward(net, batch: Dict, use_amp: bool, dev, slope: float, with_extra: bool, decode: bool = True):
    bev = batch.get("bev")
    cand = batch["cand"].float()
    extra = batch["corr_traj"].float() if with_extra else None
    with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
        return net(bev, cand, batch["status"].float(), extra=extra, decode=decode, slope=slope)


def compute_loss(net, out: Dict, batch: Dict, a, bal, step: int) -> Tuple[torch.Tensor, Dict]:
    """Full CK objective (module docstring) -> (loss, stats of floats)."""
    from navsim.agents.para_ssr.ck import losses as Lm

    Cn = C()
    st: Dict[str, float] = {}
    batch = mask_bev_ok(batch)
    cand = batch["cand"].float()
    T, K = cand.shape[:2]
    st["n_bev_bad"] = float(n_bev_bad(batch))
    logit = out["score_logit"].float()
    l_sc, per = Lm.score_bce(logit, batch["y"].float(), batch["y_ok"].bool())
    loss = a.lambda_score * l_sc
    st["score_bce"] = float(l_sc.detach())
    for k, v in (per or {}).items():
        st[f"bce_{k}"] = float(v.detach()) if torch.is_tensor(v) else float(v)
    corr_aug = a.corr_aug == "on"
    if corr_aug:
        l_cs, _ = Lm.score_bce(out["extra_score_logit"].float(), batch["y_corr"].float(), batch["y_corr_ok"].bool())
        loss = loss + a.lambda_corr_score * l_cs
        st["corr_score_bce"] = float(l_cs.detach())
    # surrogate on the GT-ok tokens (train slope)
    v0, a0 = out["ego"][0], out["ego"][1]
    idx = gt_index(batch)
    n = int(idx.numel())
    st["n_gt"] = float(n)
    l_sur_val = 0.0
    if n > 0 and a.lambda_sur:
        ref = {k: v for k, v in batch.items() if k.startswith("ref_")}
        sb = Lm.surrogate_batch_k(ref, idx, cand, batch["gt_traj"].float(), v0, a0)
        # out['corr']['raw'] is the flat T*K decode (train slope); correction_loss restricts it to sb['idx']
        l_sur, terms, nonfin = Lm.correction_loss(out["corr"]["raw"], sb, n, K, Cn.SUR_WEIGHTS, Cn.SUR_MARGINS)
        loss = loss + a.lambda_sur * l_sur
        l_sur_val = float(l_sur.detach()) if torch.is_tensor(l_sur) else float(l_sur)
        st["sur"] = l_sur_val
        st["sur_nonfinite"] = float(nonfin)
        for k, v in (terms or {}).items():
            st[f"t_{k}"] = float(v)
    # KD
    if a.kd == "on":
        kd_ok = batch["kd_ok"].bool()
        l_ks = Lm.score_kd_bce(logit, batch["kd_score_prob"].float(), kd_ok)
        if corr_aug:
            l_ks = l_ks + Lm.score_kd_bce(out["extra_score_logit"].float(), batch["kd_score_prob_corr"].float(), kd_ok)
        loss = loss + a.lambda_kd_score * l_ks
        st["kd_score"] = float(l_ks.detach())
        W = Cn.KD_CTRL_W
        l_kd, parts = Lm.ctrl_kd_l1(out["corr"]["c_lon"][..., 2:].float(), out["corr"]["e_lat"][..., 2:].float(),
                                    batch["kd_c_lon"].float(), batch["kd_e_lat"].float(), kd_ok,
                                    w_lon=W["lon"], w_lat=W["lat"])
        l_kd_val = float(l_kd.detach())
        if a.kd_ctrl_balance == "ema":
            bal.update(a.lambda_sur * l_sur_val, l_kd_val)
            w = float(bal.weight(step))
        else:
            w = float(a.kd_ctrl_weight)
        loss = loss + w * l_kd
        st["kd_ctrl"] = l_kd_val
        st["kd_w"] = w
        for k, v in (parts or {}).items():
            st[f"kd_{k}"] = float(v.detach()) if torch.is_tensor(v) else float(v)
    if a.lead_aux and "lead_logit" in out and "lead_d1" in batch:
        y = batch["lead_d1"].float()
        m = (batch["lead_has"].float() == 1) & torch.isfinite(y)
        if bool(m.any()):
            l_ld = F.binary_cross_entropy_with_logits(out["lead_logit"].float()[m], y[m])
            loss = loss + a.lambda_lead * l_ld
            st["lead_bce"] = float(l_ld.detach())
    live = (out["corr"]["c_lon"].detach() < 0).any(-1).float()
    st["live"] = float(live.mean())
    st["e_abs"] = float(out["corr"]["e_lat"].detach().abs().mean())
    st["loss"] = float(loss.detach())
    return loss, st


# ----------------------------------------------------------------------------------------------- metrics
def _bce_np(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def auc(score: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, bool)
    score = np.asarray(score, np.float64)
    if y.all() or (~y).all():
        return float("nan")
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    avg = (np.cumsum(cnt) - cnt) + (cnt + 1) / 2.0
    r = avg[inv]
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def score_metrics(prob: np.ndarray, y: np.ndarray, ok: np.ndarray, pdms: np.ndarray, v2_final: np.ndarray,
                  v2_im: np.ndarray, betas=(1.0, 0.5)) -> Dict[str, float]:
    """prob / y [N, K, 5] (CK_KEYS), ok [N, K], pdms [N, K] labels, v2_final / v2_im [N, K] -> flat metric dict:
    bce_<key>, auc_fail_<nc|dac|ttc>, pdms_v2 (cand 0), pdms_oracle (label best), pdms_a_b<beta> (CK selection with
    blend beta), on tokens whose K labels are all ok (n_sel)."""
    from navsim.agents.para_ssr.ck.select import blend, ck_final

    keys = tuple(C().CK_KEYS)
    out: Dict[str, float] = {"n_tokens": int(len(prob)), "n_cand_ok": int(ok.sum())}
    okf = ok.astype(bool)
    for j, k in enumerate(keys):
        if okf.any():
            out[f"bce_{k}"] = float(_bce_np(prob[..., j][okf], y[..., j][okf]).mean())
    for k in ("nc", "dac", "ttc"):
        j = keys.index(k)
        if okf.any():
            out[f"auc_fail_{k}"] = auc(1.0 - prob[..., j][okf], y[..., j][okf] < 1)
            out[f"fail_rate_{k}"] = float((y[..., j][okf] < 1).mean())
    sel = okf.all(1)
    out["n_sel"] = int(sel.sum())
    if sel.any():
        P, Y = pdms[sel], None
        out["pdms_v2"] = float(P[:, 0].mean())
        out["pdms_oracle"] = float(P.max(1).mean())
        ckf = np.asarray(ck_final(prob[sel].astype(np.float64), v2_im[sel].astype(np.float64)), np.float64)
        for b in betas:
            f = np.asarray(blend(v2_final[sel].astype(np.float64), ckf, float(b)), np.float64)
            ch = f.argmax(1)
            out[f"pdms_a_b{b:g}"] = float(P[np.arange(len(P)), ch].mean())
            out[f"frac_changed_b{b:g}"] = float((ch != 0).mean())
    return out


@torch.no_grad()
def evaluate(net, ds, a, dev, use_amp: bool, collate_fn) -> Dict:
    from navsim.agents.para_ssr.ck import constants as Cn
    net.eval()
    dl = make_loader(ds, a.eval_batch, 0, 0, 0, False, a.workers, collate_fn, dev.type == "cuda", drop_last=False)
    P, Y, OK, PD, VF, VI, LIVE, EA, CL = [], [], [], [], [], [], [], [], []
    t0, bad = time.time(), 0
    for batch in dl:
        batch = mask_bev_ok(to_dev(batch, dev))
        bad += n_bev_bad(batch)
        out = forward(net, batch, use_amp, dev, Cn.LON_ST_SLOPE["eval"], with_extra=False)
        P.append(torch.sigmoid(out["score_logit"].float()).cpu().numpy())
        Y.append(batch["y"].float().cpu().numpy())
        OK.append(batch["y_ok"].bool().cpu().numpy())
        PD.append(batch["y_pdms"].float().cpu().numpy())
        VF.append(batch["v2_final"].float().cpu().numpy())
        VI.append(batch["v2_im"].float().cpu().numpy())
        c = out["corr"]
        LIVE.append((c["c_lon"] < 0).any(-1).float().cpu().numpy())
        EA.append(c["e_lat"][..., 2:].abs().mean(-1).float().cpu().numpy())
        CL.append(c["c_lon"][..., 2:].mean(-1).float().cpu().numpy())
    net.train()
    if not P:
        return {"n_tokens": 0}
    cat = np.concatenate
    m = score_metrics(cat(P), cat(Y), cat(OK), cat(PD), cat(VF), cat(VI))
    m.update(corr_live=float(cat(LIVE).mean()), corr_e_abs=float(cat(EA).mean()), corr_c_lon=float(cat(CL).mean()),
             n_bev_bad=int(bad), sec=round(time.time() - t0, 1))
    return m


# ----------------------------------------------------------------------------------------------- train
def lr_lambda(warm: int, total: int):
    def f(s):
        if s < warm:
            return (s + 1) / max(1, warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm))))
    return f


def _meta(path: Path):
    return U.read_json(path, None)


def train(a, datasets: Optional[Dict] = None, collate_fn=None, norm_override=None) -> Path:
    """Run training (resumable).  datasets / collate_fn / norm_override: injection for tests."""
    from navsim.agents.para_ssr.ck.losses import EmaBalancer
    from navsim.agents.para_ssr.ck.model import save_ck

    Cn = C()
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    if a.device.startswith("cuda"):
        U.gpu_guard(a.device)
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available")
        dev = torch.device("cuda:0")
    else:
        dev = torch.device("cpu")
    run = Path(a.out_root) / a.run
    run.mkdir(parents=True, exist_ok=True)
    if a.restart:
        for f in list(run.glob("ckpt_*.pt")) + [run / x for x in ("done.json", "train_log.jsonl",
                                                                   "val_metrics.jsonl", "config.json")]:
            f.unlink(missing_ok=True)
    if (run / "done.json").is_file():
        print(f"{run} finished (done.json); --restart to retrain", flush=True)
        return run
    last = run / "ckpt_last.pt"
    if last.is_file() and not a.resume:
        raise SystemExit(f"{last} exists: pass --resume to continue or --restart to start over")
    if datasets is None:
        datasets = build_datasets(a)
    collate_fn = collate_fn or default_collate()
    train_ds = datasets["train"]
    train_tokens = getattr(train_ds, "tokens", None)
    net, init_rep = setup_model(a, run, train_tokens, norm_override)
    net = net.to(dev)
    use_amp = bool(a.amp) and dev.type == "cuda"
    ck_root = U.ck_data()
    cfg = dict(vars(a), arm=a.arm, seed=a.seed, git_head=U.git_head(), run_dir=str(run), device_resolved=str(dev),
               cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"), n_train=len(train_ds),
               evals=[(n, len(d)) for n, d in datasets["evals"]],
               packed_meta=_meta(ck_root / "packed" / a.split / "meta.json"),
               kd_meta=_meta(Path(a.kd_dir) / "meta.json") if a.kd_dir else None, init_report=init_rep,
               ck_keys=list(Cn.CK_KEYS), sur_weights=dict(Cn.SUR_WEIGHTS), sur_margins=dict(Cn.SUR_MARGINS),
               kd_ctrl_w=dict(Cn.KD_CTRL_W), lon_st_slope=dict(Cn.LON_ST_SLOPE), loss_defaults=dict(Cn.LOSS_DEFAULTS),
               param_count=int(sum(p.numel() for p in net.parameters())))
    pm = cfg["packed_meta"] or {}
    cfg["v2_ckpt_sha16"] = pm.get("v2_ckpt_sha16", pm.get("v2_sha16"))
    old = U.read_json(run / "config.json")
    if old is not None and a.resume:
        for k in ("arm", "seed", "kd", "corr_aug", "init_from", "split", "k", "lead_aux", "tokens_per_batch"):
            if old.get(k) != cfg.get(k):
                raise SystemExit(f"{run}: resume with {k}={cfg.get(k)!r} but config.json has {old.get(k)!r}")
    if old is None or a.restart:
        U.write_json(run / "config.json", cfg)
    steps_per_epoch = len(EpochBatches(len(train_ds), a.tokens_per_batch, a.seed, 0))
    if steps_per_epoch == 0:
        raise SystemExit(f"no full batch in {len(train_ds)} train tokens")
    total = steps_per_epoch * a.epochs
    if a.max_steps:
        total = min(total, a.max_steps)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(min(a.warmup_steps, max(1, total // 10))
                                                             if a.max_steps else a.warmup_steps, total))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    LD = Cn.LOSS_DEFAULTS
    bal = EmaBalancer(ratio=a.kd_ratio, m=LD["kd_ema_m"], floor=LD["kd_ema_floor"], cap=LD["kd_weight_max"],
                      start_step=LD["kd_start_step"])
    epoch, step, bstart, n_skip = 0, 0, 0, {"loss": 0, "grad": 0}
    if last.is_file():
        ck = torch.load(last, map_location="cpu")
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        if ck.get("ema"):
            bal.load_state_dict(ck["ema"])
        epoch, step, bstart = int(ck["epoch"]), int(ck["step"]), int(ck.get("batch_in_epoch", 0))
        n_skip = dict(ck.get("n_skip", n_skip))
        pending = ck.get("eval_pending")
        if bstart >= steps_per_epoch:
            epoch, bstart = epoch + 1, 0
        print(f"resumed {last}: epoch {epoch} batch {bstart} step {step}"
              + (f" (eval of epoch {pending['epoch']} pending)" if pending else ""), flush=True)
    else:
        pending = None

    def save(ep, bi, which="last", eval_pending=None):
        save_ck(run / f"ckpt_{which}.pt", net, cfg, epoch=ep, step=step, optim=opt.state_dict(),
                sched=sched.state_dict(), scaler=scaler.state_dict(), ema=bal.state_dict(), batch_in_epoch=bi,
                n_skip=dict(n_skip), eval_pending=eval_pending)

    logp, valp = run / "train_log.jsonl", run / "val_metrics.jsonl"

    def finish_epoch(pend: Dict) -> None:
        """Epoch-end (or --max-steps stop) evaluations of the weights now in ckpt_last.pt (saved with eval_pending =
        pend before this call).  Splits already in val_metrics.jsonl for (epoch, step) are skipped, so a run killed
        mid-evaluation redoes only the missing ones on --resume; then ckpt_last.pt is re-saved without the marker
        and, at a finished epoch, copied to ckpt_ep<e>.pt."""
        done_splits = set()
        if valp.is_file():
            for line in valp.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("epoch") == pend["epoch"] and r.get("step") == pend["step"]:
                        done_splits.add(r.get("split"))
        for name, ds in datasets["evals"]:
            if name in done_splits:
                continue
            m = evaluate(net, ds, a, dev, use_amp, collate_fn)
            # checked before logging: a stopped run re-evaluates this split on --resume once the cache is fixed
            check_bev_bad(int(m.get("n_bev_bad", 0)), int(m.get("n_tokens", 0)), f"eval {name}", force=True)
            U.log_jsonl(valp, dict(epoch=pend["epoch"], step=pend["step"], split=name,
                                   partial_epoch=bool(pend["partial"]), **m))
        save(pend["save_epoch"], pend["save_bi"])
        if not pend["partial"]:
            shutil.copyfile(run / "ckpt_last.pt", run / f"ckpt_ep{pend['epoch']}.pt")

    if pending:
        finish_epoch(pending)
    with_extra = a.corr_aug == "on"
    slope = Cn.LON_ST_SLOPE["train"]
    stop = step >= total
    t_run = time.time()
    bad_run, tok_run = 0, 0          # unreadable-BEV tokens / tokens seen by this process (check_bev_bad)
    while epoch < a.epochs and not stop:
        net.train()
        dl = make_loader(train_ds, a.tokens_per_batch, a.seed, epoch, bstart, True, a.workers, collate_fn,
                         dev.type == "cuda")
        t0, t_win, n_win, agg, nb = time.time(), time.time(), 0, {}, 0
        bi = bstart
        for batch in dl:
            batch = to_dev(batch, dev)
            out = forward(net, batch, use_amp, dev, slope, with_extra)
            loss, st = compute_loss(net, out, batch, a, bal, step)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                n_skip["loss"] += 1
                st["skipped"] = 1.0
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip if a.clip > 0 else float("inf"))
                st["grad_norm"] = float(gn)
                if not math.isfinite(float(gn)):
                    n_skip["grad"] += 1
                    st["skipped"] = 1.0
                    opt.zero_grad(set_to_none=True)
                    scaler.update()
                else:
                    scaler.step(opt)
                    scaler.update()
            sched.step()
            step += 1
            bi += 1
            nb += 1
            T, K = batch["cand"].shape[:2]
            n_win += T * K * (2 if with_extra else 1)
            bad_run += int(st.get("n_bev_bad", 0))
            tok_run += int(T)
            for k, v in st.items():
                agg[k] = agg.get(k, 0.0) + v
            if step % a.log_every == 0 or step == total:
                dt = time.time() - t_win
                rec = dict(kind="step", epoch=epoch, step=step, batch=bi, lr=sched.get_last_lr()[0],
                           cand_per_s=round(n_win / max(dt, 1e-6), 1), n_skip_loss=n_skip["loss"],
                           n_skip_grad=n_skip["grad"], n_bev_bad_run=bad_run, n_tok_run=tok_run,
                           **{k: round(v, 6) for k, v in st.items()})
                if dev.type == "cuda":
                    rec["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
                U.log_jsonl(logp, rec)
                t_win, n_win = time.time(), 0
                check_bev_bad(bad_run, tok_run, f"train epoch {epoch} step {step}")
            if a.ckpt_every and step % a.ckpt_every == 0:
                save(epoch, bi)
            if step >= total:
                stop = True
                break
        ep_rec = dict(kind="epoch", epoch=epoch, step=step, sec=round(time.time() - t0, 1),
                      n_bev_bad=int(agg.get("n_bev_bad", 0)), n_tok=int(nb * a.tokens_per_batch),
                      train={k: round(v / max(nb, 1), 6) for k, v in agg.items()})
        U.log_jsonl(logp, ep_rec)
        check_bev_bad(bad_run, tok_run, f"train epoch {epoch}", force=True)
        finished_epoch = bi >= steps_per_epoch
        # resume point (start of the next epoch, or here at a --max-steps stop) saved with an eval-pending marker
        # BEFORE evaluating: a kill during the evaluations re-runs them on --resume (finish_epoch) instead of
        # silently skipping this epoch's val_metrics.jsonl records
        pend = dict(epoch=epoch, step=step, partial=not finished_epoch,
                    save_epoch=epoch + 1 if finished_epoch else epoch, save_bi=0 if finished_epoch else bi)
        save(pend["save_epoch"], pend["save_bi"], eval_pending=pend)
        finish_epoch(pend)
        epoch, bstart = (epoch + 1, 0) if finished_epoch else (epoch, bi)
    summ = dict(run=a.run, arm=a.arm, steps=step, epochs_done=epoch, epochs=a.epochs, total_steps=total,
                max_steps=a.max_steps, n_skip=n_skip, n_bev_bad_last_session=bad_run, n_tok_last_session=tok_run,
                sec=round(time.time() - t_run, 1), finished=U.now())
    U.write_json(run / "done.json", summ)
    print(f"done {run}: {summ}", flush=True)
    return run


def main(argv=None):
    a = get_parser().parse_args(argv)
    if a.workers > 16:
        raise SystemExit("--workers <= 16 per process (48 per agent in total)")
    resolve_args(a)
    train(a)


if __name__ == "__main__":
    main()
