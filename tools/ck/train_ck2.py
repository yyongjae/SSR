#!/usr/bin/env python
"""Train one CK2 teacher (r34-free; user design 2026-10-08): DET (arm T, BEVFusion t0 BEV cache) or MAP (arm M, ReSMap
BEV cache), from scratch, on raw anchors + rule variants with official labels (EP target per --ep-target).  Single GPU
or 2-GPU DDP (torchrun).

  # 1 GPU
  cd /workspace/yongjae/SSR-ck2 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/yongjae/SSR-ck2 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 /venv/ssr/bin/python tools/ck/train_ck2.py --arm T --run-name ck2T --resume
  # 2 GPUs (one process per GPU; --standalone picks a free rendezvous port)
  cd /workspace/yongjae/SSR-ck2 && CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=/workspace/yongjae/SSR-ck2 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 /venv/ssr/bin/torchrun --standalone --nproc_per_node=2 tools/ck/train_ck2.py --arm T --run-name ck2T \
    --resume
  Under torchrun spell the run option --run-name (alias of --run): torchrun's own argparse (Python 3.9) scans every
  argument, the script's included, for abbreviations of its options and aborts on '--run' ("ambiguous option: --run
  could match --run-path, --run_path").  Alternatively put '--' before the script path.

Data (tools/ck/data/ck2_dataset.CK2Dataset; per training token, deterministic in (seed, epoch, token))
  cand      K = 32 raw anchors from AnchorSampler (8 near + 8 mid + 16 pass/fail far), re-drawn every epoch (sampler
            epoch = training epoch), random slot order; labels = labels/<split>/raw256 (official, 9 columns).
  var_traj  --n-var (default 32) of the token's valid non-identity variants of the fixed near+mid-16 subset
            (ck2/labels/<split>/<--var-name>, 5 variants x 16 anchors = 80), re-drawn every epoch, random order.
  Model inputs are only bev, cand, status, var_traj (MODEL_INPUT_KEYS); GT trajectory / sampler metadata are not.
  --ep-target (user 2026-10-08 ~20:10 KST) = which EP the score head's EP key is trained / evaluated on:
    official   (default; old runs, bit-identical) the official EP column, which is 0 whenever NC, DAC or DDC fails
    decoupled  EP = clip(r / max(p, r), 0, 1) if max(p, r) > 5 m else 1 (r = raw_progress, p = pdm_progress_eff:
               the official EP without the NC * DAC * DDC factor; == official EP wherever NC = DAC = DDC = 1;
               non-finite -> NaN -> candidate not ok).  navsim.agents.para_ssr.ck.ep_target.ck_targets is the ONE
               conversion of the 9 label columns into the 5 CK targets: anchor y, variant var_y, the score prior
               (ck2_label_prior) and every evaluation set (r34 top-16 'cand' labels, val / train-eval raw + variants),
               so EP BCE / MAE / corr are measured against the trained target.  Unchanged: y_pdms / var_pdms and every
               selection PDMS (official), the AnchorSampler pass / fail strata (official NC / DAC / TTC / C, EP unused).
    Recorded in config.json (ep_target, ep_target_rule) and part of the --resume consistency check (a config.json
    without the key = official).

Model: navsim.agents.para_ssr.ck.model.CKNet (CKTrunk + score head + lateral head), init_from none by default.
  The longitudinal head is REMOVED: trunk.lon_head is zeroed and frozen (z_lon == 0 for every loader of the
  checkpoint) and the decode uses z_lon = 0 explicitly; trunk.gate_head (never in any loss) is frozen too, so DDP
  needs no find_unused_parameters.  Lateral head (w_lat) is trained by the surrogate and applied after selection.

Loss (outside autocast, f32):
  L = lambda_score * score_bce(cat[anchors, variants])        [5 keys nc dac ep ttc comfort, unweighted BCE, official
                                                               labels (EP per --ep-target); anchors / variants also
                                                               logged separately]
    + lambda_sur * correction_loss(decode(anchors, z_lon = 0, w_lat, slope 0.1))   [v1 GT surrogate on GT-ok tokens,
                                                               anchors of --sur-groups (default all 32);
                                                               col/dac/ttc/prog/cmf/mod terms as train_ck, reported]
    + 0 * sum(w_lat)                                          [keeps the lateral head in every DDP graph]
  No KD, no lon head loss, no prior / calibration correction (anc_pi / anc_w / var_pi stay in the batch for later).
Optimiser (as train_ck): AdamW (trainable parameters), linear warmup --warmup-steps then cosine to 0 over
  epochs x steps_per_epoch, fp16 autocast + GradScaler, clip --clip; a step whose loss (any rank) or grad norm is
  non-finite is skipped and counted.  Score-head bias = logit(expected training label mean) (--score-prior).

DDP / batch: --tokens-per-batch is the GLOBAL batch (default 8 = train_ck's single-GPU batch); each of the W ranks
  takes tokens_per_batch / W tokens of every global batch.  The global batch sequence (seeded permutation per epoch,
  train_ck.EpochBatches) does not depend on W, so 1 GPU and 2 GPUs run the same optimisation (same LR, same schedule,
  same steps; DDP averages the per-rank mean losses' gradients = the global mean when the per-rank masks have equal
  counts) and resume works across W.  The LR is NOT rescaled automatically: raising --tokens-per-batch with --lr
  unchanged lowers the per-sample step size (linear rule: lr x B / 8 if wanted).  Each rank seeds torch / numpy with
  seed + rank (the data pipeline itself is keyed by token, not by rank).  Rank 0 alone writes files; loss statistics
  are gathered over ranks (counts summed, means averaged); evaluations are sharded over ranks and gathered.

Evaluations (every --eval-every epochs, at the last epoch and at a --max-steps stop) -> val_metrics.jsonl ('split',
'kind' fields):
  <val>            kind r34: navtrain_val r34 top-16 candidates (CKDataset 'cand' labels, train_ck val path): BCE /
                   fail-AUC per key, pass AUC, EP MAE / corr, selection PDMS (pdms_a_b1, pdms_a_b0.5 with v2 im / blend
                   as train_ck; pdms_ck_noim without v2 im; pdms_ck_plugin with the 'plugin' weight set of
                   select.SEL_W_SETS, descriptive), lateral |e|.  DET only (no ReSMap cache for navtrain_val).
  <val>_raw        kind raw: the same val tokens, sampler epoch 0 (K = 32) + --eval-n-var variants: per-key BCE / AUC
                   for anchors and variants, pass AUC per anchor group / variant type, selection among the 32 anchors
                   (GT-informed set: relative use only), variant-vs-parent pair ranking accuracy per key and PDMS.
  <split>_train_eval[_raw]  the same two kinds on a log-stratified --train-eval-rows subset of the trained rows
                   (MAP's only evaluation; train-vs-val gap of DET).
Outputs --out-root/<run>/ (default CK_DATA/ck2/train): config.json, norm.npz | norm_map.npz, ckpt_last.pt (every
  --ckpt-every steps and every epoch end; mid-epoch resume continues the same batch order), ckpt_ep<e>.pt,
  ckpt_best.pt (+ best.json) when --best-metric improves (default T: navtrain_val:pdms_a_b1:max, M: none),
  train_log.jsonl, val_metrics.jsonl, done.json.  ckpt files hold {'model' (CKNet state dict, no 'module.' prefix),
  'cfg', ...}: model.load_ck(run, 'last' | 'best' | 'ep<e>') and model.build_ck(init_from=run) load them.
  Epoch-end evaluation is crash-safe as in train_ck (eval_pending marker in ckpt_last.pt, missing splits redone).
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
import datetime  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from typing import Dict, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from tools.ck import ckutil as U  # noqa: E402
from tools.ck import train_ck as TC  # noqa: E402

ARMS = ("T", "M")
NORM_FILE = TC.NORM_FILE
TRAIN_EVAL_NAME = TC.TRAIN_EVAL_NAME
RAW_SUFFIX = "_raw"
FROZEN_MODULES = ("trunk.lon_head", "trunk.gate_head")
SUM_KEYS = ("n_gt", "n_bev_bad", "n_anc_ok", "n_var_ok")
DEFAULT_BEST = {"T": "navtrain_val:pdms_a_b1:max", "M": "none"}
SUR_GROUPS = ("all", "near", "near_mid")
SUR_GROUP_MAX = {"all": None, "near": 0, "near_mid": 1}       # anchor_sampler GROUP_NEAR 0 / GROUP_MID 1 / GROUP_STRAT 2
FAIL_KEYS = ("nc", "dac", "ttc", "comfort")


def C():
    from navsim.agents.para_ssr.ck import constants
    return constants


# ----------------------------------------------------------------------------------------------- args
def get_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="train a CK2 teacher (T / M) from scratch, 1 GPU or 2-GPU DDP",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--run", "--run-name", dest="run", required=True,
                    help="run name (use the spelling --run-name under torchrun: see the module docstring)")
    ap.add_argument("--out-root", default=None, help="default CK_DATA/ck2/train")
    ap.add_argument("--split", default="navtrain_train")
    ap.add_argument("--val-split", default=None, help="navtrain_val | none (default: T navtrain_val, M none)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4, help="not rescaled with the batch (see the module docstring)")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--tokens-per-batch", type=int, default=8, help="GLOBAL tokens per optimiser step (all ranks)")
    ap.add_argument("--n-var", type=int, default=80,
                    help="variant candidates per token and step (0 = none; 80 = all, user 2026-10-08)")
    ap.add_argument("--var-name", default="var_separate_sampler16_accstraight",
                    help="variant label set (accstraight: a+0.5 with the straight path extension, user 2026-10-08)")
    ap.add_argument("--sampler-seed", type=int, default=0, help="AnchorSampler seed (variants were built with 0)")
    ap.add_argument("--workers", type=int, default=8, help="DataLoader workers per rank")
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default="none", help="'none' (scratch, default) or a CK / CK2 run dir")
    ap.add_argument("--norm-from", default="compute",
                    help="'compute' (z-score over 2048 train tokens, seed 0) or a run dir holding norm.npz / "
                         "norm_map.npz (a data statistic, not learned weights)")
    ap.add_argument("--lambda-score", type=float, default=None)
    ap.add_argument("--lambda-sur", type=float, default=None)
    ap.add_argument("--score-keys", default="nc,dac,ep,ttc,comfort",
                    help="comma list of CK keys that get the score BCE (specialist teachers, user 2026-10-08: "
                         "T nc,ttc,ep,comfort / M dac,ep); other keys' heads are still evaluated but not trained")
    ap.add_argument("--sur-groups", choices=SUR_GROUPS, default="all",
                    help="anchor groups in the lateral surrogate: all 32 (default) | near (8) | near_mid (16).  "
                         "Measured on 48 train tokens with the identity decode: surrogate 1.92 (all) / 0.87 (near_mid) / "
                         "0.19 (near) vs 0.088 on v2 r34 top-16 (the old teacher's candidates); the far strat anchors "
                         "dominate (2.97, mostly cmf / dac / ttc)")
    ap.add_argument("--ep-target", choices=("official", "decoupled"), default="official",
                    help="EP target of the score head (module docstring): official (default, old behaviour) | "
                         "decoupled (official EP without the NC * DAC * DDC factor; user 2026-10-08)")
    ap.add_argument("--score-prior", type=int, default=1, help="score-head bias = logit(expected train label mean)")
    ap.add_argument("--prior-rows", type=int, default=4000)
    ap.add_argument("--train-eval-rows", type=int, default=8192)
    ap.add_argument("--val-raw", type=int, default=1, help="also evaluate the val raw anchors + variants")
    ap.add_argument("--eval-n-var", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=1, help="epochs between evaluations (last epoch always)")
    ap.add_argument("--best-metric", default=None, help="'split:metric:max|min' or 'none' (default by arm)")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--limit-tokens", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--eval-batch", type=int, default=8, help="tokens per eval batch per rank")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=2000, help="mid-epoch ckpt_last.pt every N steps (0 = epoch only)")
    ap.add_argument("--device", default="cuda", help="cuda (CUDA_VISIBLE_DEVICES: 1-2 GPUs of 0-3) | cpu")
    ap.add_argument("--dist-timeout-min", type=float, default=60.0)
    ap.add_argument("--resume", action="store_true", help="continue from ckpt_last.pt if present")
    ap.add_argument("--restart", action="store_true", help="wipe the run dir outputs and start over")
    return ap


def resolve_args(a):
    LD = dict(C().LOSS_DEFAULTS)
    if a.out_root is None:
        a.out_root = str(U.ck_data() / "ck2" / "train")
    if a.val_split is None:
        a.val_split = "none" if a.arm == "M" else "navtrain_val"
    if a.best_metric is None:
        a.best_metric = DEFAULT_BEST[a.arm] if a.val_split != "none" else "none"
    if a.lambda_score is None:
        a.lambda_score = float(LD["lambda_score"])
    if a.lambda_sur is None:
        a.lambda_sur = float(LD["lambda_sur"])
    if a.init_from in (None, ""):
        a.init_from = "none"
    from navsim.agents.para_ssr.ck.constants import CK_KEYS
    ks = [k.strip() for k in str(a.score_keys).split(",") if k.strip()]
    if not ks or len(set(ks)) != len(ks) or any(k not in CK_KEYS for k in ks):
        raise SystemExit(f"--score-keys must be a non-empty subset of {list(CK_KEYS)}, got {ks}")
    a.score_keys = [k for k in CK_KEYS if k in ks]                       # canonical order (config.json list)
    from navsim.agents.para_ssr.ck.ep_target import check_ep_target
    try:
        a.ep_target = check_ep_target(getattr(a, "ep_target", "official"))
    except ValueError as e:
        raise SystemExit(f"--ep-target: {e}") from None
    if a.arm == "M" and a.val_split != "none":
        raise SystemExit("arm M: no ReSMap cache for the val split (--val-split none)")
    if a.workers > 16:
        raise SystemExit("--workers <= 16 per rank")
    parse_best(a.best_metric)
    return a


def parse_best(s: Optional[str]) -> Optional[Tuple[str, str, str]]:
    if not s or s == "none":
        return None
    p = s.split(":")
    if len(p) != 3 or p[2] not in ("max", "min"):
        raise SystemExit(f"--best-metric {s!r}: expected 'split:metric:max|min' or 'none'")
    return p[0], p[1], p[2]


# ----------------------------------------------------------------------------------------------- distributed
class Dist:
    """Process-group context from the torchrun environment (WORLD_SIZE / RANK / LOCAL_RANK); world 1 = no group."""

    def __init__(self, device: str = "cuda", timeout_min: float = 60.0):
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local = int(os.environ.get("LOCAL_RANK", "0"))
        self.is0 = self.rank == 0
        self.cuda = str(device).startswith("cuda")
        self._group = False
        if self.cuda:
            U.gpu_guard_ddp(device, self.world)
            if not torch.cuda.is_available():
                raise SystemExit("CUDA requested but not available")
            self.device = torch.device("cuda", self.local)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        if self.world > 1:
            kw = dict(device_id=self.device) if self.cuda else {}
            dist.init_process_group("nccl" if self.cuda else "gloo",
                                    timeout=datetime.timedelta(minutes=float(timeout_min)), **kw)
            self._group = True
            assert dist.get_world_size() == self.world and dist.get_rank() == self.rank

    def barrier(self) -> None:
        if self._group:
            dist.barrier()

    def all_gather(self, obj) -> List:
        if not self._group:
            return [obj]
        out = [None] * self.world
        dist.all_gather_object(out, obj)
        return out

    def bcast(self, obj):
        if not self._group:
            return obj
        lst = [obj if self.is0 else None]
        dist.broadcast_object_list(lst, src=0)
        return lst[0]

    def all_min(self, x: float) -> float:
        if not self._group:
            return float(x)
        t = torch.tensor([float(x)], dtype=torch.float64, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return float(t.item())

    def all_sum(self, xs: Sequence[float]) -> List[float]:
        if not self._group:
            return [float(x) for x in xs]
        t = torch.tensor([float(x) for x in xs], dtype=torch.float64, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return [float(v) for v in t.tolist()]

    def close(self, clean: bool = True) -> None:
        """Clean exit: barrier + destroy.  After an exception no collective is attempted (the other ranks may sit in
        one): the process exits and torchrun stops the remaining workers."""
        if self._group and dist.is_initialized():
            if clean:
                dist.barrier()
                dist.destroy_process_group()
            self._group = False


def merge_stats(parts: List[Dict[str, float]], denom: Optional[List[float]] = None) -> Dict[str, float]:
    """Combine per-rank stat dicts: SUM_KEYS summed, everything else averaged over the ranks that report it."""
    keys = sorted({k for p in parts for k in p})
    out = {}
    for k in keys:
        vals = [float(p[k]) for p in parts if k in p]
        out[k] = float(np.sum(vals)) if k in SUM_KEYS else float(np.mean(vals))
    return out


class DistEpochBatches(torch.utils.data.Sampler):
    """Rank slice of train_ck.EpochBatches: the seeded global batch order of an epoch (rng(seed * 100003 + epoch),
    drop_last), global batch b -> this rank takes b[rank * B / W : (rank + 1) * B / W]; starts at global batch `start`.
    Equivalent to a DistributedSampler with set_epoch, but the global batches are independent of W (1 vs 2 GPUs see
    the same token groups per step) and a mid-epoch resume restarts at an exact batch."""

    def __init__(self, n: int, global_bs: int, seed: int, epoch: int, start: int, rank: int, world: int):
        if global_bs % world:
            raise ValueError(f"global batch {global_bs} not divisible by world size {world}")
        self.base = TC.EpochBatches(n, global_bs, seed, epoch, 0, True, True)
        self.start, self.rank, self.world = int(start), int(rank), int(world)
        self.local = global_bs // world

    def batches(self) -> List[np.ndarray]:
        lo = self.rank * self.local
        return [b[lo:lo + self.local] for b in self.base.batches()]

    def __iter__(self):
        for b in self.batches()[self.start:]:
            yield [int(i) for i in b]

    def __len__(self):
        return max(0, len(self.base.batches()) - self.start)


# ----------------------------------------------------------------------------------------------- data
def build_datasets(a) -> Dict:
    """-> {'train': CK2Dataset, 'evals': [(name, kind in {'r34', 'raw'}, dataset)]}."""
    from tools.ck.data.ck2_dataset import CK2Dataset
    from tools.ck.data.ck_dataset import CKDataset

    ep = getattr(a, "ep_target", "official")
    ck2 = dict(var_name=a.var_name, seed=a.seed, sampler_seed=a.sampler_seed, ep_target=ep)
    train = CK2Dataset(a.split, bev=a.arm, n_var=a.n_var, gt=True, limit=a.limit_tokens or 0, **ck2)
    evals = []
    if a.val_split and a.val_split != "none":
        evals.append((a.val_split, "r34", CKDataset(a.val_split, bev=a.arm, k=C().K_CAND, labels="cand", gt=False,
                                                    limit=a.val_limit or 0, ep_target=ep)))
        if a.val_raw:
            evals.append((a.val_split + RAW_SUFFIX, "raw",
                          CK2Dataset(a.val_split, bev=a.arm, n_var=a.eval_n_var, gt=False, epoch=0,
                                     limit=a.val_limit or 0, **ck2)))
    if a.train_eval_rows:
        tdf = TC.packed_tokens(a.split)
        if tdf is not None:
            pool = np.asarray(train.rows, np.int64)
            n_eval = a.train_eval_rows if not a.val_limit else min(a.train_eval_rows, a.val_limit)
            rows = pool[U.log_stratified_rows(tdf["log"].values[pool], n_eval, seed=0)]
            name = TRAIN_EVAL_NAME.format(a.split)
            evals.append((name, "r34", CKDataset(a.split, bev=a.arm, k=C().K_CAND, labels="cand", gt=False,
                                                 rows=rows, ep_target=ep)))
            evals.append((name + RAW_SUFFIX, "raw", CK2Dataset(a.split, bev=a.arm, n_var=a.eval_n_var, gt=False,
                                                               epoch=0, rows=rows, **ck2)))
    return {"train": train, "evals": evals}


def collate():
    from tools.ck.data.ck_dataset import collate_ck
    return collate_ck


def make_loader(ds, batch_sampler=None, batch_size: int = 1, workers: int = 0, pin: bool = False, collate_fn=None):
    from torch.utils.data import DataLoader
    kw = dict(batch_sampler=batch_sampler) if batch_sampler is not None else dict(batch_size=int(batch_size),
                                                                                    shuffle=False, drop_last=False)
    return DataLoader(ds, collate_fn=collate_fn or collate(), num_workers=int(workers), pin_memory=pin,
                      prefetch_factor=4 if workers > 0 else None, persistent_workers=False, **kw)


# ----------------------------------------------------------------------------------------------- model
def freeze_unused(net) -> List[str]:
    """CK2: zero and freeze the longitudinal head (z_lon == 0 for every user of the checkpoint) and freeze the gate
    head (never in a loss).  -> frozen parameter names."""
    lh = net.trunk.lon_head
    with torch.no_grad():
        lh[-1].weight.zero_()
        lh[-1].bias.zero_()
    names = []
    for mod_name in FROZEN_MODULES:
        mod = net.get_submodule(mod_name)
        for n, p in mod.named_parameters():
            p.requires_grad_(False)
            names.append(f"{mod_name}.{n}")
    return names


def trainable(net) -> List[torch.nn.Parameter]:
    return [p for p in net.parameters() if p.requires_grad]


def setup_model(a, run: Path, train_ds, D: Dist, norm_override=None):
    """-> (CKNet with frozen lon / gate heads, init report).  Rank 0 writes the norm file; every rank builds the net
    from it (DDP then broadcasts rank 0's parameters, incl. the score prior)."""
    from navsim.agents.para_ssr.ck.model import build_ck
    from navsim.agents.para_ssr.refiner.adapters import load_norm, save_norm
    from tools.ck.data.ck2_dataset import ck2_label_prior

    dst = run / NORM_FILE[a.arm]
    if D.is0 and not dst.is_file():
        if norm_override is not None:
            save_norm(dst, norm_override[0], norm_override[1], {"source": "override (tests)"})
        elif a.norm_from and a.norm_from != "compute":
            src = Path(a.norm_from) / NORM_FILE[a.arm]
            if not src.is_file():
                raise SystemExit(f"--norm-from {a.norm_from}: {src} missing")
            shutil.copyfile(src, dst)
        else:
            TC.compute_norm(a.arm, list(getattr(train_ds, "tokens")), run)
    D.barrier()
    mean, std, _ = load_norm(dst)
    init = None if a.init_from in (None, "", "none") else a.init_from
    net = build_ck(a.arm, seed=a.seed, norm=(mean, std), init_from=init, lead_aux=False)
    rep = getattr(net, "init_report", None)
    rep = dict(rep) if isinstance(rep, dict) else {"report": str(rep)}
    if a.score_prior and rep.get("kind") != "ck" and D.is0:
        prior = ck2_label_prior(train_ds, a.prior_rows) if hasattr(train_ds, "labels_item") else None
        if prior is not None:
            net.set_score_prior(prior)
            rep["score_prior"] = [float(x) for x in prior]
    rep["frozen"] = freeze_unused(net)
    return net, rep


# ----------------------------------------------------------------------------------------------- forward / loss
def mask_bev_ok(batch: Dict) -> Dict:
    """Tokens whose BEV could not be read drop out of every loss / metric mask (anchors, variants, surrogate)."""
    if "bev_ok" not in batch:
        return batch
    ok = batch["bev_ok"].bool().reshape(-1)
    if bool(ok.all()):
        return batch
    b = dict(batch)
    for k in ("y_ok", "var_ok"):
        if k in b:
            b[k] = b[k].bool() & ok[:, None]
    if "ref_gt_ok" in b:
        b["ref_gt_ok"] = b["ref_gt_ok"].bool() & ok
    return b


def forward(net, batch: Dict, use_amp: bool, dev, with_var: bool) -> Dict:
    """CKNet forward on the model inputs only (bev, cand, status, var_traj); no decode (done with z_lon = 0)."""
    extra = batch["var_traj"].float() if with_var and "var_traj" in batch else None
    with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
        return net(batch.get("bev"), batch["cand"].float(), batch["status"].float(), extra=extra, decode=False)


def lateral_decode(cand: torch.Tensor, w_lat: torch.Tensor, v0: torch.Tensor, slope: float) -> Dict:
    """decode(cand, z_lon = 0, w_lat): the learned longitudinal head is removed in CK2."""
    from navsim.agents.para_ssr.ck.correct import correct
    w = w_lat.float()
    return correct(cand.float(), torch.zeros_like(w), w, v0, slope)


def score_bce_keys(logit: torch.Tensor, y: torch.Tensor, ok: torch.Tensor, keys) -> Tuple[torch.Tensor, Dict[str, float]]:
    """losses.score_bce restricted to `keys` (specialist teachers): the loss is the mean of the selected keys' per-key
    BCE; every key's BCE is still returned for logging.  keys = all CK_KEYS reproduces losses.score_bce exactly."""
    import torch.nn.functional as F
    from navsim.agents.para_ssr.ck import losses as Lm
    from navsim.agents.para_ssr.ck.constants import CK_KEYS
    m = Lm._mask(ok, y)
    yy = torch.where(m[..., None], y.float(), torch.zeros_like(y, dtype=torch.float32)).clamp(0.0, 1.0)
    per = F.binary_cross_entropy_with_logits(logit.float(), yy, reduction="none")
    pk = Lm._key_mean(per, m)
    ki = [CK_KEYS.index(k) for k in (keys or CK_KEYS)]
    return pk[ki].mean(), {k: float(pk[i].detach()) for i, k in enumerate(CK_KEYS)}


def compute_loss(out: Dict, corr: Dict, batch: Dict, a) -> Tuple[torch.Tensor, Dict[str, float]]:
    """CK2 teacher objective (module docstring) -> (loss, stats of floats)."""
    from navsim.agents.para_ssr.ck import losses as Lm

    Cn = C()
    st: Dict[str, float] = {}
    batch = mask_bev_ok(batch)
    cand = batch["cand"].float()
    T, K = cand.shape[:2]
    st["n_bev_bad"] = float(TC.n_bev_bad(batch))
    la, ya, oka = out["score_logit"].float(), batch["y"].float(), batch["y_ok"].bool()
    has_var = "extra_score_logit" in out and "var_y" in batch
    if has_var:
        lv, yv, okv = out["extra_score_logit"].float(), batch["var_y"].float(), batch["var_ok"].bool()
        l_sc, per = score_bce_keys(torch.cat([la, lv], 1), torch.cat([ya, yv], 1), torch.cat([oka, okv], 1),
                                   getattr(a, "score_keys", None))
    else:
        l_sc, per = score_bce_keys(la, ya, oka, getattr(a, "score_keys", None))
    loss = a.lambda_score * l_sc
    st["score_bce"] = float(l_sc.detach())
    st.update({f"bce_{k}": float(v) for k, v in per.items()})
    with torch.no_grad():
        l_a, per_a = Lm.score_bce(la.detach(), ya, oka)
        st["anc_bce"] = float(l_a)
        st.update({f"anc_bce_{k}": float(v) for k, v in per_a.items()})
        st["n_anc_ok"] = float((oka & torch.isfinite(ya).all(-1)).sum())
        if has_var:
            l_v, per_v = Lm.score_bce(lv.detach(), yv, okv)
            st["var_bce"] = float(l_v)
            st.update({f"var_bce_{k}": float(v) for k, v in per_v.items()})
            st["n_var_ok"] = float((okv & torch.isfinite(yv).all(-1)).sum())
    # lateral surrogate on the GT-ok tokens (anchors decoded with z_lon = 0, train slope)
    v0, a0 = out["ego"][0], out["ego"][1]
    idx = TC.gt_index(batch)
    n = int(idx.numel())
    st["n_gt"] = float(n)
    if n > 0 and a.lambda_sur:
        ref = {k: v for k, v in batch.items() if k.startswith("ref_")}
        sb = Lm.surrogate_batch_k(ref, idx, cand, batch["gt_traj"].float(), v0, a0)
        gmax = SUR_GROUP_MAX[getattr(a, "sur_groups", "all")]
        valid = None
        if gmax is not None:
            if "anc_group" not in batch:
                raise ValueError(f"--sur-groups {a.sur_groups}: batch has no anc_group")
            valid = batch["anc_group"][idx].to(cand.device).long() <= gmax          # [n, K] metadata mask only
        l_sur, terms, nonfin = Lm.correction_loss(corr["raw"], sb, n, K, Cn.SUR_WEIGHTS, Cn.SUR_MARGINS, valid=valid)
        loss = loss + a.lambda_sur * l_sur
        st["sur"] = float(l_sur.detach()) if torch.is_tensor(l_sur) else float(l_sur)
        st["sur_nonfinite"] = float(nonfin)
        st.update({f"t_{k}": float(v) for k, v in (terms or {}).items()})
    loss = loss + 0.0 * out["w_lat"].float().sum()          # lateral head always in the graph (DDP)
    st["e_abs"] = float(corr["e_lat"].detach()[..., 2:].abs().mean())
    st["w_lat_abs"] = float(out["w_lat"].detach().float().abs().mean())
    st["z_lon_absmax"] = float(out["z_lon"].detach().float().abs().max())
    st["loss"] = float(loss.detach())
    return loss, st


def dummy_backward(out: Dict) -> None:
    """DDP: when the step is skipped (non-finite loss on some rank) every rank still runs one backward over the same
    graph (zero-weighted outputs), so the reducer stays in step; the gradients are discarded."""
    terms = [out[k].float().nan_to_num(0.0, 0.0, 0.0).sum() * 0.0 for k in ("score_logit", "extra_score_logit",
                                                                            "w_lat") if k in out]
    if terms:
        torch.stack(terms).sum().backward()


# ----------------------------------------------------------------------------------------------- metrics
def _auc_fail(prob: np.ndarray, y: np.ndarray) -> float:
    return TC.auc(1.0 - prob, y < 1)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a.astype(np.float64), b.astype(np.float64))[0, 1])


def key_metrics(prob: np.ndarray, y: np.ndarray, ok: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """prob / y [M, 5] (CK_KEYS), ok [M] -> bce_<k>, auc_fail_<k> / fail_rate_<k> (nc, dac, ttc, comfort; fail =
    label < 1), auc_pass (p_pass = product of the 4 fail-key probabilities vs pass label), pass_rate, ep_mae, ep_corr,
    auc_mean (mean of the finite fail AUCs)."""
    keys = tuple(C().CK_KEYS)
    ok = np.asarray(ok, bool) & np.isfinite(y).all(-1)
    out: Dict[str, float] = {f"{prefix}n_cand": int(ok.sum())}
    if not ok.any():
        return out
    P, Y = prob[ok].astype(np.float64), y[ok].astype(np.float64)
    for j, k in enumerate(keys):
        out[f"{prefix}bce_{k}"] = float(TC._bce_np(P[:, j], Y[:, j]).mean())
    aucs = []
    for k in FAIL_KEYS:
        j = keys.index(k)
        out[f"{prefix}auc_fail_{k}"] = _auc_fail(P[:, j], Y[:, j])
        out[f"{prefix}fail_rate_{k}"] = float((Y[:, j] < 1).mean())
        if math.isfinite(out[f"{prefix}auc_fail_{k}"]):
            aucs.append(out[f"{prefix}auc_fail_{k}"])
    fi = [keys.index(k) for k in FAIL_KEYS]
    ypass = (Y[:, fi] >= 1 - 1e-6).all(1)
    out[f"{prefix}auc_pass"] = TC.auc(P[:, fi].prod(1), ypass)
    out[f"{prefix}pass_rate"] = float(ypass.mean())
    j = keys.index("ep")
    out[f"{prefix}ep_mae"] = float(np.abs(P[:, j] - Y[:, j]).mean())
    out[f"{prefix}ep_corr"] = _corr(P[:, j], Y[:, j])
    out[f"{prefix}auc_mean"] = float(np.mean(aucs)) if aucs else float("nan")
    return out


def ck_score_noim(prob: np.ndarray, w="default") -> np.ndarray:
    """select.ck_final without the v2 imitation term (im = 1 -> constant); w = a select.SEL_W_SETS name or 4 weights."""
    from navsim.agents.para_ssr.ck.select import ck_final
    p = prob.astype(np.float64)
    return np.asarray(ck_final(p, np.ones(p.shape[:-1]), w), np.float64)


def r34_metrics(R: Dict[str, np.ndarray]) -> Dict[str, float]:
    """r34 top-16 candidates: train_ck.score_metrics (+ comfort AUC, pass AUC, EP, selection without v2 im)."""
    P, Y, OK, PD = R["prob"], R["y"], R["y_ok"].astype(bool), R["y_pdms"]
    m = TC.score_metrics(P, Y, OK, PD, R["v2_final"], R["v2_im"])
    km = key_metrics(P.reshape(-1, 5), Y.reshape(-1, 5), OK.reshape(-1))
    for k in ("auc_fail_comfort", "fail_rate_comfort", "auc_pass", "pass_rate", "ep_mae", "ep_corr", "auc_mean"):
        if k in km:
            m[k] = km[k]
    sel = OK.all(1)
    if sel.any():
        ch = ck_score_noim(P[sel]).argmax(1)
        m["pdms_ck_noim"] = float(PD[sel][np.arange(int(sel.sum())), ch].mean())
        m["frac_changed_noim"] = float((ch != 0).mean())
        ch = ck_score_noim(P[sel], "plugin").argmax(1)               # descriptive: w_im 0, NC 1, DAC 1, rest 1
        m["pdms_ck_plugin"] = float(PD[sel][np.arange(int(sel.sum())), ch].mean())
    m["corr_e_abs"] = float(R["e_abs"].mean()) if len(R["e_abs"]) else float("nan")
    return m


def raw_metrics(R: Dict[str, np.ndarray], var_names: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Raw anchors (sampler epoch 0) + variants: per-key metrics for each, pass AUC per anchor group / variant type,
    selection among the anchors (GT-informed set), variant-vs-parent pair ranking accuracy."""
    from navsim.agents.para_ssr.ck.anchor_sampler import GROUP_NAMES
    keys = tuple(C().CK_KEYS)
    Pa, Ya, OKa, PDa, G = R["prob"], R["y"], R["y_ok"].astype(bool), R["y_pdms"], R["anc_group"]
    out: Dict[str, float] = {"n_tokens": int(len(Pa))}
    out.update(key_metrics(Pa.reshape(-1, 5), Ya.reshape(-1, 5), OKa.reshape(-1), "anc_"))
    fi = [keys.index(k) for k in FAIL_KEYS]
    for g, gn in enumerate(GROUP_NAMES):
        m = (G == g) & OKa
        if m.any():
            out[f"anc_auc_pass_{gn}"] = TC.auc(Pa[m][:, fi].prod(1), (Ya[m][:, fi] >= 1 - 1e-6).all(1))
            out[f"anc_pass_rate_{gn}"] = float((Ya[m][:, fi] >= 1 - 1e-6).all(1).mean())
    sel = OKa.all(1)
    out["anc_n_sel"] = int(sel.sum())
    if sel.any():
        ch = ck_score_noim(Pa[sel]).argmax(1)
        P = PDa[sel]
        out["anc_pdms_sel"] = float(P[np.arange(len(P)), ch].mean())
        out["anc_pdms_oracle"] = float(P.max(1).mean())
        out["anc_pdms_mean"] = float(P.mean())
    if "var_prob" in R and len(R["var_prob"]):
        Pv, Yv, OKv, PDv = R["var_prob"], R["var_y"], R["var_ok"].astype(bool), R["var_pdms"]
        VV, VA, AI = R["var_v"].astype(np.int64), R["var_anchor"], R["anc_idx"]
        out.update(key_metrics(Pv.reshape(-1, 5), Yv.reshape(-1, 5), OKv.reshape(-1), "var_"))
        for v in np.unique(VV):
            m = (VV == v) & OKv
            nm = var_names[v] if var_names is not None and v < len(var_names) else f"v{v}"
            if m.any():
                out[f"var_auc_pass_{nm}"] = TC.auc(Pv[m][:, fi].prod(1), (Yv[m][:, fi] >= 1 - 1e-6).all(1))
                out[f"var_pass_rate_{nm}"] = float((Yv[m][:, fi] >= 1 - 1e-6).all(1).mean())
        # parent = the identity anchor of the variant in the sampled anchor set (epoch-0 near+mid contains it)
        eq = AI[:, None, :] == VA[:, :, None]                                  # [T, V, K]
        has = eq.any(-1)
        j = eq.argmax(-1)
        t = np.arange(len(Pv))[:, None]
        Pp, Yp, OKp, PDp = Pa[t, j], Ya[t, j], OKa[t, j], PDa[t, j]
        pair = has & OKv & OKp
        out["pair_n"] = int(pair.sum())
        out["pair_frac_parent_found"] = float(has.mean()) if has.size else float("nan")
        for jk, k in enumerate(keys):
            dy, dp = Yv[..., jk] - Yp[..., jk], Pv[..., jk] - Pp[..., jk]
            m = pair & (np.abs(dy) > 1e-6)
            out[f"pair_n_{k}"] = int(m.sum())
            if m.any():
                out[f"pair_acc_{k}"] = float((np.sign(dp[m]) == np.sign(dy[m])).mean())
        dy = PDv - PDp
        df = ck_score_noim(Pv) - ck_score_noim(Pp)
        m = pair & (np.abs(dy) > 1e-6)
        out["pair_n_pdms"] = int(m.sum())
        if m.any():
            out["pair_acc_pdms"] = float((np.sign(df[m]) == np.sign(dy[m])).mean())
    out["corr_e_abs"] = float(R["e_abs"].mean()) if len(R["e_abs"]) else float("nan")
    return out


@torch.no_grad()
def run_eval(net, ds, kind: str, a, dev, use_amp: bool, D: Dist, collate_fn=None) -> Dict[str, np.ndarray]:
    """Sharded evaluation (rank r takes the r-th contiguous slice of the rows) -> arrays of all ranks (rank order)."""
    net.eval()
    shard = np.array_split(np.arange(len(ds)), D.world)[D.rank]
    sub = torch.utils.data.Subset(ds, shard.tolist())
    dl = make_loader(sub, batch_size=a.eval_batch, workers=a.workers if len(shard) else 0, pin=dev.type == "cuda",
                     collate_fn=collate_fn)
    keep = (("prob", "y", "y_ok", "y_pdms", "v2_final", "v2_im") if kind == "r34" else
            ("prob", "y", "y_ok", "y_pdms", "anc_group", "anc_idx", "var_prob", "var_y", "var_ok", "var_pdms",
             "var_v", "var_anchor"))
    acc: Dict[str, list] = {k: [] for k in keep + ("e_abs",)}
    bad, ntok = 0, 0
    slope = C().LON_ST_SLOPE["eval"]
    if len(shard):
        for batch in dl:
            batch = mask_bev_ok(TC.to_dev(batch, dev))
            bad += TC.n_bev_bad(batch)
            ntok += int(batch["cand"].shape[0])
            with_var = kind == "raw" and "var_traj" in batch
            out = forward(net, batch, use_amp, dev, with_var)
            corr = lateral_decode(batch["cand"], out["w_lat"], out["ego"][0], slope)
            acc["prob"].append(torch.sigmoid(out["score_logit"].float()).cpu().numpy())
            acc["e_abs"].append(corr["e_lat"][..., 2:].abs().mean(-1).float().cpu().numpy().reshape(-1))
            if with_var:
                acc["var_prob"].append(torch.sigmoid(out["extra_score_logit"].float()).cpu().numpy())
            for k in keep:
                if k in ("prob", "var_prob"):
                    continue
                if k in batch:
                    acc[k].append(batch[k].cpu().numpy())
    net.train()
    mine = {k: (np.concatenate(v) if v else None) for k, v in acc.items()}
    mine["_bad"], mine["_ntok"] = bad, ntok
    parts = D.all_gather(mine)
    R: Dict[str, np.ndarray] = {}
    for k in acc:
        arrs = [p[k] for p in parts if p[k] is not None]
        if arrs:
            R[k] = np.concatenate(arrs)
    R["n_bev_bad"] = np.asarray(sum(int(p["_bad"]) for p in parts))
    R["n_tok"] = np.asarray(sum(int(p["_ntok"]) for p in parts))
    return R


def eval_metrics(kind: str, R: Dict[str, np.ndarray], var_names=None) -> Dict[str, float]:
    if "prob" not in R:
        return {"n_tokens": 0, "n_bev_bad": int(R.get("n_bev_bad", 0))}
    m = r34_metrics(R) if kind == "r34" else raw_metrics(R, var_names)
    m["n_tokens"] = int(R["n_tok"])
    m["n_bev_bad"] = int(R["n_bev_bad"])
    return m


# ----------------------------------------------------------------------------------------------- train
def _meta(path: Path):
    return U.read_json(path, None)


def train(a, datasets: Optional[Dict] = None, collate_fn=None, norm_override=None) -> Path:
    """Run training (resumable; 1 process or one process per GPU under torchrun).  datasets / collate_fn /
    norm_override: injection for tests."""
    D = Dist(a.device, a.dist_timeout_min)
    ok = False
    try:
        run = _train(a, D, datasets, collate_fn, norm_override)
        ok = True
        return run
    finally:
        D.close(clean=ok)


def _train(a, D: Dist, datasets, collate_fn, norm_override) -> Path:
    from torch.nn.parallel import DistributedDataParallel as DDP

    from navsim.agents.para_ssr.ck.ep_target import EP_RULE
    from navsim.agents.para_ssr.ck.model import save_ck

    Cn = C()
    a.ep_target = getattr(a, "ep_target", "official")
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    torch.manual_seed(a.seed + D.rank)
    np.random.seed((a.seed + D.rank) % 2 ** 32)
    dev = D.device
    if a.tokens_per_batch % D.world:
        raise SystemExit(f"--tokens-per-batch {a.tokens_per_batch} not divisible by world size {D.world}")
    b_local = a.tokens_per_batch // D.world
    run = Path(a.out_root) / a.run
    if D.is0:
        run.mkdir(parents=True, exist_ok=True)
        if a.restart:
            for f in list(run.glob("ckpt_*.pt")) + [run / x for x in ("done.json", "train_log.jsonl", "best.json",
                                                                       "val_metrics.jsonl", "config.json")]:
                f.unlink(missing_ok=True)
    D.barrier()
    if (run / "done.json").is_file():
        if D.is0:
            print(f"{run} finished (done.json); --restart to retrain", flush=True)
        return run
    last = run / "ckpt_last.pt"
    if last.is_file() and not a.resume:
        raise SystemExit(f"{last} exists: pass --resume to continue or --restart to start over")
    if datasets is None:
        datasets = build_datasets(a)
    collate_fn = collate_fn or collate()
    train_ds = datasets["train"]
    evals = datasets["evals"]
    net, init_rep = setup_model(a, run, train_ds, D, norm_override)
    net = net.to(dev)
    params = trainable(net)
    use_amp = bool(a.amp) and dev.type == "cuda"
    ck_root = U.ck_data()
    var_names = (getattr(train_ds, "var_meta", None) or {}).get("names")
    cfg = dict(vars(a), arm=a.arm, seed=a.seed, trainer="train_ck2", git_head=U.git_head(), run_dir=str(run),
               device_resolved=str(dev), cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"), world_size=D.world,
               tokens_per_rank=b_local, global_batch=a.tokens_per_batch,
               lr_note="lr is not rescaled with the batch; global batches are world-size independent",
               n_train=len(train_ds), evals=[(n, k, len(d)) for n, k, d in evals],
               train_data=train_ds.describe() if hasattr(train_ds, "describe") else None,
               raw256_meta=_meta(ck_root / "labels" / a.split / "raw256" / "meta.json"),
               init_report=init_rep, ck_keys=list(Cn.CK_KEYS), sur_weights=dict(Cn.SUR_WEIGHTS),
               sur_margins=dict(Cn.SUR_MARGINS), lon_st_slope=dict(Cn.LON_ST_SLOPE),
               lon_head="removed (zeroed + frozen; decode with z_lon = 0)", kd="off", score_weighting="none",
               ep_target_rule=(EP_RULE if a.ep_target == "decoupled" else "official EP label column"),
               score_hidden=256, lead_aux=0, param_count=int(sum(p.numel() for p in net.parameters())),
               param_count_trainable=int(sum(p.numel() for p in params)))
    old = U.read_json(run / "config.json")
    if old is not None and a.resume:
        old_default = {"sur_groups": "all", "score_keys": list(Cn.CK_KEYS), "ep_target": "official"}
        for k in ("arm", "seed", "split", "n_var", "var_name", "sampler_seed", "tokens_per_batch", "init_from",
                  "lambda_score", "lambda_sur", "sur_groups", "score_keys", "ep_target"):
            ov = old.get(k, old_default.get(k))
            if ov != cfg.get(k):
                raise SystemExit(f"{run}: resume with {k}={cfg.get(k)!r} but config.json has {ov!r}")
    if D.is0 and (old is None or a.restart):
        U.write_json(run / "config.json", cfg)
    steps_per_epoch = len(TC.EpochBatches(len(train_ds), a.tokens_per_batch, a.seed, 0))
    if steps_per_epoch == 0:
        raise SystemExit(f"no full batch in {len(train_ds)} train tokens")
    total = steps_per_epoch * a.epochs
    if a.max_steps:
        total = min(total, a.max_steps)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, TC.lr_lambda(min(a.warmup_steps, max(1, total // 10))
                                                                if a.max_steps else a.warmup_steps, total))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_spec = parse_best(a.best_metric)
    epoch, step, bstart, n_skip = 0, 0, 0, {"loss": 0, "grad": 0}
    best: Dict = {}
    pending = None
    if last.is_file():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        epoch, step, bstart = int(ck["epoch"]), int(ck["step"]), int(ck.get("batch_in_epoch", 0))
        n_skip = dict(ck.get("n_skip", n_skip))
        best = dict(ck.get("best") or {})
        pending = ck.get("eval_pending")
        if bstart >= steps_per_epoch:
            epoch, bstart = epoch + 1, 0
        if D.is0:
            print(f"resumed {last}: epoch {epoch} batch {bstart} step {step}"
                  + (f" (eval of epoch {pending['epoch']} pending)" if pending else ""), flush=True)
    model = (DDP(net, device_ids=[dev.index] if dev.type == "cuda" else None, broadcast_buffers=True)
             if D.world > 1 else net)

    def save(ep, bi, which="last", eval_pending=None):
        if D.is0:
            save_ck(run / f"ckpt_{which}.pt", net, cfg, epoch=ep, step=step, optim=opt.state_dict(),
                    sched=sched.state_dict(), scaler=scaler.state_dict(), ema=None, batch_in_epoch=bi,
                    n_skip=dict(n_skip), eval_pending=eval_pending, best=dict(best), world_size=D.world)

    logp, valp = run / "train_log.jsonl", run / "val_metrics.jsonl"

    def recorded(ep: int, st: int) -> Dict[str, Dict]:
        out = {}
        if valp.is_file():
            for line in valp.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("epoch") == ep and r.get("step") == st:
                        out[r.get("split")] = r
        return out

    def finish_epoch(pend: Dict) -> None:
        """Evaluations of the weights now in ckpt_last.pt (saved with eval_pending = pend), sharded over ranks;
        splits already in val_metrics.jsonl for (epoch, step) are skipped.  Then ckpt_last.pt is re-saved without
        the marker, copied to ckpt_ep<e>.pt at a finished epoch and to ckpt_best.pt when --best-metric improved."""
        nonlocal best
        do_eval = pend.get("eval", True)
        done = D.bcast(recorded(pend["epoch"], pend["step"]) if D.is0 else None) if do_eval else {}
        res = dict(done)
        if do_eval:
            for name, kind, ds in evals:
                if name in done:
                    continue
                R = run_eval(net, ds, kind, a, dev, use_amp, D, collate_fn)
                m = eval_metrics(kind, R, var_names)
                TC.check_bev_bad(int(m.get("n_bev_bad", 0)), int(m.get("n_tokens", 0)), f"eval {name}", force=True)
                if D.is0:
                    U.log_jsonl(valp, dict(epoch=pend["epoch"], step=pend["step"], split=name, kind=kind,
                                           partial_epoch=bool(pend["partial"]), **m))
                res[name] = m
        improved = False
        if best_spec is not None and best_spec[0] in res:
            v = res[best_spec[0]].get(best_spec[1])
            if v is not None and math.isfinite(float(v)):
                v = float(v)
                if not best or (v > best["value"] if best_spec[2] == "max" else v < best["value"]):
                    best = dict(metric=a.best_metric, value=v, epoch=pend["epoch"], step=pend["step"],
                                partial_epoch=bool(pend["partial"]))
                    improved = True
        save(pend["save_epoch"], pend["save_bi"])
        if D.is0:
            if not pend["partial"]:
                shutil.copyfile(run / "ckpt_last.pt", run / f"ckpt_ep{pend['epoch']}.pt")
            if improved:
                shutil.copyfile(run / "ckpt_last.pt", run / "ckpt_best.pt")
                U.write_json(run / "best.json", best)
        D.barrier()

    if pending:
        finish_epoch(pending)
    slope = Cn.LON_ST_SLOPE["train"]
    with_var = a.n_var > 0
    stop = step >= total
    t_run = time.time()
    bad_run, tok_run = 0, 0
    while epoch < a.epochs and not stop:
        model.train()
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        sampler = DistEpochBatches(len(train_ds), a.tokens_per_batch, a.seed, epoch, bstart, D.rank, D.world)
        dl = make_loader(train_ds, batch_sampler=sampler, workers=a.workers, pin=dev.type == "cuda",
                         collate_fn=collate_fn)
        t0, t_win, n_win, agg, nb = time.time(), time.time(), 0, {}, 0
        bi = bstart
        for batch in dl:
            batch = TC.to_dev(batch, dev)
            out = forward(model, batch, use_amp, dev, with_var)
            corr = lateral_decode(batch["cand"], out["w_lat"], out["ego"][0], slope)
            loss, st = compute_loss(out, corr, batch, a)
            opt.zero_grad(set_to_none=True)
            if D.all_min(1.0 if bool(torch.isfinite(loss)) else 0.0) < 1.0:
                n_skip["loss"] += 1
                st["skipped"] = 1.0
                if D.world > 1:
                    dummy_backward(out)
                    opt.zero_grad(set_to_none=True)
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(params, a.clip if a.clip > 0 else float("inf"))
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
            T = int(batch["cand"].shape[0])
            n_win += T * (int(batch["cand"].shape[1]) + (int(batch["var_traj"].shape[1]) if with_var and
                                                         "var_traj" in batch else 0))
            bad_run += int(st.get("n_bev_bad", 0))
            tok_run += T
            for k, v in st.items():
                agg[k] = agg.get(k, 0.0) + v
            if step % a.log_every == 0 or step == total:
                g = merge_stats(D.all_gather(st))
                bad_all, tok_all, n_all = D.all_sum([bad_run, tok_run, n_win])
                dt = time.time() - t_win
                if D.is0:
                    rec = dict(kind="step", epoch=epoch, step=step, batch=bi, lr=sched.get_last_lr()[0],
                               cand_per_s=round(n_all / max(dt, 1e-6), 1), n_skip_loss=n_skip["loss"],
                               n_skip_grad=n_skip["grad"], n_bev_bad_run=int(bad_all), n_tok_run=int(tok_all),
                               world=D.world, **{k: round(v, 6) for k, v in g.items()})
                    if dev.type == "cuda":
                        rec["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
                    U.log_jsonl(logp, rec)
                t_win, n_win = time.time(), 0
                TC.check_bev_bad(int(bad_all), int(tok_all), f"train epoch {epoch} step {step}")
            if a.ckpt_every and step % a.ckpt_every == 0:
                save(epoch, bi)
            if step >= total:
                stop = True
                break
        parts = D.all_gather(agg)
        tot = {}
        for p in parts:
            for k, v in p.items():
                tot[k] = tot.get(k, 0.0) + v
        nb_all = max(nb, 1)
        mean = {k: (v / nb_all if k in SUM_KEYS else v / (nb_all * D.world)) for k, v in tot.items()}
        bad_all, tok_all = D.all_sum([bad_run, tok_run])
        if D.is0:
            U.log_jsonl(logp, dict(kind="epoch", epoch=epoch, step=step, sec=round(time.time() - t0, 1),
                                   n_bev_bad=int(tot.get("n_bev_bad", 0)), n_tok=int(nb * a.tokens_per_batch),
                                   world=D.world, train={k: round(v, 6) for k, v in mean.items()}))
        TC.check_bev_bad(int(bad_all), int(tok_all), f"train epoch {epoch}", force=True)
        finished_epoch = bi >= steps_per_epoch
        do_eval = (not finished_epoch) or stop or epoch + 1 >= a.epochs or (epoch + 1) % max(1, a.eval_every) == 0
        pend = dict(epoch=epoch, step=step, partial=not finished_epoch, eval=do_eval,
                    save_epoch=epoch + 1 if finished_epoch else epoch, save_bi=0 if finished_epoch else bi)
        save(pend["save_epoch"], pend["save_bi"], eval_pending=pend)
        D.barrier()
        finish_epoch(pend)
        epoch, bstart = (epoch + 1, 0) if finished_epoch else (epoch, bi)
    bad_all, tok_all = D.all_sum([bad_run, tok_run])
    summ = dict(run=a.run, arm=a.arm, steps=step, epochs_done=epoch, epochs=a.epochs, total_steps=total,
                max_steps=a.max_steps, n_skip=n_skip, n_bev_bad_last_session=int(bad_all),
                n_tok_last_session=int(tok_all),
                world=D.world, best=best or None, sec=round(time.time() - t_run, 1), finished=U.now())
    if D.is0:
        U.write_json(run / "done.json", summ)
        print(f"done {run}: {summ}", flush=True)
    return run


def main(argv=None):
    a = get_parser().parse_args(argv)
    resolve_args(a)
    train(a)


if __name__ == "__main__":
    main()
