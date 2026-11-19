#!/usr/bin/env python
"""Train one FPD-structured privileged teacher refiner (ck_e2e2.arch 'fpd'; FPD-Drive arXiv 2610.13232: residual DiT +
candidate scorer, navsim/agents/para_ssr/ck/fpd_net.FPDRefinerNet) on one cached teacher BEV, offline, in three stages.
  arm T = DET (BEVFusion t0 cache; the e2e kd_bev_0)  |  arm M = MAP (ReSMap front64 cache; the e2e kd_bev_1)
The finished run dir is what fpd_net.load_fpd_teacher reads (ck_e2e2.fpd.teacher_det_run / teacher_map_run).

  # every stage: 1 GPU, or torchrun with 2 / 4 GPUs (the global batch is --tokens-per-batch whatever the GPU count)
  cd $REPO && CUDA_VISIBLE_DEVICES=0,1,2,3 CK_BEV64=1 CK_TEACHER_MAX_GPUS=4 PYTHONPATH=$REPO OMP_NUM_THREADS=1 \
    $SSR_TORCHRUN --standalone --nproc_per_node=4 tools/ck/fpd/train_fpd_teacher.py --stage dit --arm T --resume
  ... --stage bank --arm T --resume                         (GPU sampling of the DiT bank, both splits)
  CUDA_VISIBLE_DEVICES= $SSR_PY tools/ck/fpd/label_fpd_bank.py --run-dir <run dir> --workers 48   (CPU, official)
  ... --stage scorer --arm T --resume
  tools/release/setup/run_fpd_teachers.sh runs this recipe for both teachers.  CK_BEV64=1 is required (the teachers
  read the 64 m BEVs f16 [256, 100, 100], the tensor the e2e student hands its teachers).

Data: tools/ck/data/ck2_dataset.CK2Dataset(split, bev=arm, n_var=0): per token the teacher BEV (f16 [256, 100, 100],
  fed as .float(), no z-score: SceneTokens has its own LayerNorm), the AnchorSampler's K = 32 raw anchors of the epoch's
  draw (8 near + 8 mid + 16 pass / fail far, random slot order; official raw256 labels y / y_ok), status [8], the GT
  trajectory [8, 3].  Model inputs: bev, cand / bank (candidate poses), status; GT and labels only in the losses.
  Parents = the 16 near + mid anchors of the item (anc_group <= 1).

Stages (run dir default ${SSR_TEACHER_CACHE}/refiner_fpd/{det|map}, --run-dir)
  dit     FPD-Drive S1 on the GT target: J = 1 target (the GT trajectory), Hungarian-matched to its nearest parent
          (FPD Eq. 5), loss = lambda_diff L_diff + lambda_traj L_traj = fpd_net.dit_loss (FPD Eq. 6; t / epsilon from
          token-keyed generators: stream 'fpd.train', keyed by (token, epoch, global step)).  The scorer is frozen (no
          loss reaches it).  AdamW lr 2e-4, weight decay 0.01, linear warmup + cosine, clip 1.0, fp32 (--amp 1 = the
          fp16 autocast + GradScaler of train_ck2), 10 epochs, 16 tokens per global batch.  -> run_dir/dit/.
  bank    with the finished DiT (eval mode, fp32): for every token of --bank-splits (default navtrain_train and
          navtrain_val), the 16 near + mid anchors of the sampler's epoch-0 draw in GT-rank order, 2 DDIM draws each
          (--steps) from fpd_net.token_noise(token, (16, 2, 8, 3), seed, epoch -1, 'fpd.teacher_bank') -> 32 bank
          trajectories per token (column c = 2 k + d) -> run_dir/bank/<split>/ (tools/ck/fpd/bank.py layout).  Rows are
          sharded over the ranks; resumable per row (rows whose BEV was unreadable are tried again; more than 1 % of
          a split unreadable refuses sample_done.json).  The official labels are then computed on CPU by
          tools/ck/fpd/label_fpd_bank.py (process pool, score_trajectories.score_token on the navtrain metric cache).
  scorer  from the finished dit checkpoint, the whole net (scene tokens + DiT + scorer) for 10 epochs: per token ONE
          candidate set of 64 = the 32 raw anchors of the epoch's draw (official raw256 labels) + the token's 32 bank
          trajectories (official bank labels) through net.score (self-attention across the set); loss = lambda_score
          BCE on the arm's score keys only (T: nc, ttc, ep, comfort; M: dac, ep, the CK2 teacher split; the other keys'
          heads are frozen) + the stage-dit DiT loss (keeps the shared scene tokens good for the DiT; its noise keys use
          epoch + 1000, so they differ from stage dit's).  Score-head bias = logit(training label mean) (bank.py
          bank_label_prior: epoch-0 anchors + bank).  Tokens whose bank is not labelled are left out (BankDataset);
          each split's bank must come from the finished dit checkpoint and this run's dataset rows (bank.check_bank).
          At the end the teacher run dir is written: ckpt_last.pt (fpd_net.save_fpd_teacher: {'state_dict', 'config'}),
          config.json {trainer 'train_fpd_teacher', arm, score_keys, ep_target 'official', seed, steps, grid, stage
          'scorer', split, init_from 'none', bev_shape, recipe, stages, init_report.score_prior, ...}, done.json.

DDP / batches / resume: as tools/ck/train_ck2.py (its Dist, DistEpochBatches, merge_stats, make_loader, key_metrics are
  reused): global batches independent of the world size (train_ck.EpochBatches(seed, epoch)), rank r takes its slice,
  per-rank mean losses (DDP averages the gradients), non-finite loss / grad steps skipped and counted, rank 0 writes.
  Each training stage dir holds config.json, ckpt_last.pt (model + optimiser state; every --ckpt-every steps and every
  epoch end; mid-epoch resume restarts at the exact batch), ckpt_ep<e>.pt (weights only), train_log.jsonl,
  val_metrics.jsonl, done.json.  --resume continues a stage; a resume whose recipe keys differ from the stage's
  config.json is refused.  Epoch-end evaluations are crash-safe (eval_pending marker, as train_ck2).
Evaluations (every --eval-every epochs and the last; --val-split, --val-limit = first N rows) -> val_metrics.jsonl:
  dit / scorer   val DiT losses at fixed noise (epoch -1), ADE of the parents / bank draws to the GT (parent_min_ade,
                 bank_min_ade, match_ade, match_draw_min_ade, frac_match_improved, draw_shift)
  scorer         + per-key BCE / fail AUROC / pass AUROC / EP MAE / corr of the scorer on the 64-candidate sets
                 (train_ck2.key_metrics: all, anc_ (raw anchors), bank_), auc_keys_mean = mean fail AUROC of the
                 arm's fail keys.
"""
from __future__ import annotations

import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[3])   # repo root of this worktree
if sys.path[0] != CK:
    sys.path.insert(0, CK)
import navsim  # noqa: E402

assert navsim.__file__.startswith(CK), navsim.__file__

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from typing import Dict, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from navsim.agents.para_ssr.ck import fpd_net as FN  # noqa: E402
from tools.ck import ckutil as U  # noqa: E402
from tools.ck import train_ck as TC  # noqa: E402
from tools.ck import train_ck2 as T2  # noqa: E402
from tools.ck.fpd import bank as BK  # noqa: E402

ARMS = ("T", "M")
STAGES = ("dit", "bank", "scorer")
TEACHER_OF_ARM = {arm: name for name, arm in FN.FPD_TEACHER_ARMS.items()}         # T -> det, M -> map
DEFAULT_SCORE_KEYS = {"T": ("nc", "ttc", "ep", "comfort"), "M": ("dac", "ep")}   # the CK2 teacher split
NOISE_EPOCH_OFFSET = {"dit": 0, "scorer": 1000}        # dit_loss noise epoch = training epoch + offset
VAL_NOISE_EPOCH = -1
EP_TARGET = "official"
SUM_KEYS = ("n_bev_bad", "n_pairs", "n_anc_ok", "n_bank_ok")
RESUME_KEYS = {"dit": ("arm", "seed", "split", "sampler_seed", "tokens_per_batch", "steps", "grid", "lambda_diff",
                       "lambda_traj"),
               "scorer": ("arm", "seed", "split", "sampler_seed", "tokens_per_batch", "steps", "grid", "lambda_diff",
                          "lambda_traj", "lambda_score", "score_keys", "ep_target", "dit_ckpt_sha16",
                          "bank_traj_sha16")}
SAMPLE_PLAN_KEYS = ("split", "arm", "n_packed", "n_rows", "rows_sha16", "dit_ckpt_sha16", "seed", "sampler_seed",
                    "sampler_epoch", "stream", "noise_epoch", "steps", "grid", "n_parents", "n_draws")
FINAL_CONFIG_KEYS = ("trainer", "arm", "score_keys", "ep_target", "seed", "steps", "grid", "stage")
FINALIZE_KEYS = ("arm", "seed", "split", "sampler_seed", "steps", "grid", "score_keys", "ep_target")


# ----------------------------------------------------------------------------------------------- args
def get_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="train an FPD-structured teacher refiner (stages dit / bank / scorer)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--stage", choices=STAGES, required=True)
    ap.add_argument("--arm", choices=ARMS, required=True, help="T = DET (BEVFusion cache), M = MAP (ReSMap cache)")
    ap.add_argument("--run-dir", default=None, help="default ${SSR_TEACHER_CACHE}/refiner_fpd/{det|map} (by arm)")
    ap.add_argument("--split", default="navtrain_train")
    ap.add_argument("--val-split", default="navtrain_val", help="navtrain_val | none")
    ap.add_argument("--bank-splits", default=None, help="stage bank: comma list (default --split and --val-split)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4, help="not rescaled with the batch (FPD-Drive: 2e-4, batch 16)")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--tokens-per-batch", type=int, default=16, help="GLOBAL tokens per optimiser step (all ranks)")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--amp", type=int, default=0, choices=(0, 1),
                    help="0 = fp32 (default); 1 = fp16 autocast + GradScaler (the train_ck2 choice), CUDA only")
    ap.add_argument("--seed", type=int, default=0, help="net construction + every token-keyed noise stream")
    ap.add_argument("--sampler-seed", type=int, default=0, help="AnchorSampler seed")
    ap.add_argument("--workers", type=int, default=8, help="DataLoader workers per rank")
    ap.add_argument("--score-keys", default=None,
                    help="comma list of CK keys the scorer BCE trains (default T nc,ttc,ep,comfort / M dac,ep)")
    ap.add_argument("--lambda-diff", type=float, default=1.0)
    ap.add_argument("--lambda-traj", type=float, default=1.0)
    ap.add_argument("--lambda-score", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=FN.DDIM_STEPS, help="DDIM steps (bank, evaluations, the teacher)")
    ap.add_argument("--grid", type=int, default=FN.SCENE_GRID, help="scene token grid (grid x grid tokens)")
    ap.add_argument("--score-prior", type=int, default=1, choices=(0, 1),
                    help="stage scorer: score-head bias = logit(training label mean)")
    ap.add_argument("--prior-rows", type=int, default=4000)
    ap.add_argument("--sample-batch", type=int, default=32, help="stage bank: tokens per sampling batch per rank")
    ap.add_argument("--eval-every", type=int, default=2, help="epochs between evaluations (the last one always)")
    ap.add_argument("--eval-batch", type=int, default=16, help="tokens per eval batch per rank")
    ap.add_argument("--val-limit", type=int, default=0, help="first N rows of --val-split (0 = all; tests)")
    ap.add_argument("--limit-tokens", type=int, default=0, help="first N rows of --split (0 = all; tests)")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=2000, help="mid-epoch ckpt_last.pt every N steps (0 = epoch only)")
    ap.add_argument("--device", default="cuda", help="cuda (CUDA_VISIBLE_DEVICES: 1, 2 or 4 GPUs) | cpu")
    ap.add_argument("--dist-timeout-min", type=float, default=60.0)
    ap.add_argument("--resume", action="store_true", help="continue the stage from its ckpt_last.pt / partial bank")
    ap.add_argument("--restart", action="store_true",
                    help="wipe this stage's outputs first (stage bank: the bank and label files of --bank-splits)")
    return ap


def default_run_dir(arm: str) -> Path:
    from navsim.common import ssr_paths as SP
    return Path(SP.SSR_TEACHER_CACHE) / "refiner_fpd" / TEACHER_OF_ARM[arm]


def resolve_args(a):
    from navsim.agents.para_ssr.ck.constants import CK_KEYS
    rd = Path(a.run_dir) if a.run_dir else default_run_dir(a.arm)
    a.run_dir = str(rd.absolute())
    if "=" in a.run_dir:
        raise SystemExit(f"--run-dir {a.run_dir!r}: no '=' (hydra override values)")
    ks = BK.parse_list(a.score_keys) if a.score_keys else list(DEFAULT_SCORE_KEYS[a.arm])
    if not ks or len(set(ks)) != len(ks) or any(k not in CK_KEYS for k in ks):
        raise SystemExit(f"--score-keys must be a non-empty subset of {list(CK_KEYS)}, got {ks}")
    a.score_keys = [k for k in CK_KEYS if k in ks]                       # canonical order (config.json list)
    a.val_split = "none" if a.val_split in (None, "", "none", "None") else str(a.val_split)
    a.bank_splits = (BK.parse_list(a.bank_splits) if a.bank_splits else
                     [a.split] + ([a.val_split] if a.val_split != "none" else []))
    if a.split not in a.bank_splits and a.stage == "bank":
        print(f"note: --bank-splits {a.bank_splits} leaves out the training split {a.split}", flush=True)
    if not 0 <= a.workers <= 16:
        raise SystemExit("--workers in [0, 16] per rank")
    if a.tokens_per_batch < 1 or a.epochs < 1 or a.eval_batch < 1 or a.sample_batch < 1:
        raise SystemExit("--tokens-per-batch, --epochs, --eval-batch, --sample-batch must be >= 1")
    if not 1 <= a.steps <= 1000 or a.grid < 1:
        raise SystemExit(f"--steps in [1, 1000] and --grid >= 1 (got {a.steps}, {a.grid})")
    a.ep_target = EP_TARGET
    return a


# ----------------------------------------------------------------------------------------------- data / model
def require_bev64() -> None:
    from navsim.agents.para_ssr.refiner.adapters import BEV64
    if not BEV64:
        raise SystemExit("FPD teachers read the 64 m teacher BEVs (f16 [256, 100, 100], the e2e kd_bev_0 / kd_bev_1): "
                         "run every process with CK_BEV64=1")


def base_dataset(a, split: str):
    """CK2Dataset(split, bev=arm, n_var=0), sampler epoch 0 (training sets set_epoch every epoch)."""
    from tools.ck.data.ck2_dataset import CK2Dataset
    lim = BK.stage_rows_limit(split, a.split, a.val_split, a.limit_tokens, a.val_limit)
    return CK2Dataset(split, bev=a.arm, n_var=0, gt=False, epoch=0, seed=a.seed, sampler_seed=a.sampler_seed,
                      limit=lim, ep_target=EP_TARGET)


def get_base(a, split: str, datasets: Optional[Dict] = None):
    """datasets: {split: CK2Dataset-like} injection (tests); otherwise a real CK2Dataset (CK_BEV64=1 required)."""
    if datasets is not None:
        if split not in datasets:
            raise SystemExit(f"split {split!r} not among the injected datasets {sorted(datasets)}")
        return datasets[split]
    require_bev64()
    return base_dataset(a, split)


def build_net(a) -> FN.FPDRefinerNet:
    return FN.FPDRefinerNet(seed=a.seed, steps=a.steps, grid=a.grid)


def freeze_for_stage(net: FN.FPDRefinerNet, stage: str, score_keys) -> List[str]:
    """dit: the whole scorer (no loss reaches it); scorer: the heads of the keys outside score_keys (never in the
    loss; DDP needs every trainable parameter in the graph).  -> frozen parameter names."""
    from navsim.agents.para_ssr.ck.constants import CK_KEYS
    if stage == "dit":
        mods = [("scorer", net.scorer)]
    else:
        mods = [(f"scorer.heads.{i}", h) for i, (k, h) in enumerate(zip(CK_KEYS, net.scorer.heads))
                if k not in score_keys]
    names = []
    for mn, m in mods:
        for n, p in m.named_parameters():
            p.requires_grad_(False)
            names.append(f"{mn}.{n}")
    return names


def trainable(net) -> List[torch.nn.Parameter]:
    return [p for p in net.parameters() if p.requires_grad]


class StageModel(nn.Module):
    """The training forward of stage dit / scorer, run through DDP (checkpoints hold self.net.state_dict())."""

    def __init__(self, net: FN.FPDRefinerNet, seed: int):
        super().__init__()
        self.net = net
        self.seed = int(seed)

    def forward(self, bev, cand, anc_group, gt, tmask, status, keys, noise_epoch: int, step: int, bank=None):
        ctx = self.net.scene(bev)
        parents = BK.near_mid(cand.float(), anc_group)
        gt = gt.float()
        fin = torch.isfinite(gt).reshape(gt.shape[0], -1).all(1)
        gt = torch.where(fin[:, None, None], gt, torch.zeros_like(gt))        # masked rows stay finite (no NaN * 0)
        tm = tmask.bool().reshape(-1) & fin
        l_diff, l_traj, st = FN.dit_loss(self.net, parents, gt[:, None], tm[:, None], ctx, status.float(), keys,
                                         self.seed, int(noise_epoch), int(step))
        out = {"l_diff": l_diff, "l_traj": l_traj, "n_pairs": st["fpd/n_pairs"], "res_abs": st["fpd/res_abs"]}
        if bank is not None:
            out["logit"] = self.net.score(torch.cat([cand.float(), bank.float()], 1), ctx, status.float())
        return out


def bev_ok_of(batch: Dict) -> torch.Tensor:
    if "bev_ok" in batch:
        return batch["bev_ok"].bool().reshape(-1)
    s = batch["status"]
    return torch.ones(s.shape[0], dtype=torch.bool, device=s.device)


def merge_stats(parts: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """per-rank stat dicts -> SUM_KEYS summed, everything else averaged over the ranks that report it."""
    keys = sorted({k for p in parts for k in p})
    out = {}
    for k in keys:
        vals = [float(p[k]) for p in parts if k in p]
        out[k] = float(np.sum(vals)) if k in SUM_KEYS else float(np.mean(vals))
    return out


def forward(model, batch: Dict, stage: str, epoch: int, step: int, use_amp: bool, dev) -> Dict:
    with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
        return model(bev=batch["bev"], cand=batch["cand"], anc_group=batch["anc_group"], gt=batch["gt_traj"],
                     tmask=bev_ok_of(batch), status=batch["status"], keys=list(batch["tokens"]),
                     noise_epoch=epoch + NOISE_EPOCH_OFFSET[stage], step=step,
                     bank=batch["bank_traj"] if stage == "scorer" else None)


def compute_loss(out: Dict, batch: Dict, a, stage: str):
    """-> (loss, stats of floats): lambda_diff L_diff + lambda_traj L_traj (+ lambda_score score BCE on the arm's
    keys over the 64-candidate set, stage scorer).  Tokens with an unreadable BEV are out of every term."""
    st: Dict[str, float] = {"n_bev_bad": float(TC.n_bev_bad(batch))}
    l_diff, l_traj = out["l_diff"].float(), out["l_traj"].float()
    loss = a.lambda_diff * l_diff + a.lambda_traj * l_traj
    st.update(l_diff=float(l_diff.detach()), l_traj=float(l_traj.detach()), n_pairs=float(out["n_pairs"]),
              res_abs=float(out["res_abs"]))
    if stage == "scorer":
        ok_tok = bev_ok_of(batch)
        lg = out["logit"].float()
        ya, yb = batch["y"].float(), batch["bank_y"].float()
        oka = batch["y_ok"].bool() & ok_tok[:, None]
        okb = batch["bank_ok"].bool() & ok_tok[:, None]
        K = ya.shape[1]
        l_sc, per = T2.score_bce_keys(lg, torch.cat([ya, yb], 1), torch.cat([oka, okb], 1), a.score_keys)
        loss = loss + a.lambda_score * l_sc
        st["score_bce"] = float(l_sc.detach())
        st.update({f"bce_{k}": float(v) for k, v in per.items()})
        with torch.no_grad():
            la, _ = T2.score_bce_keys(lg[:, :K].detach(), ya, oka, a.score_keys)
            lb, _ = T2.score_bce_keys(lg[:, K:].detach(), yb, okb, a.score_keys)
        st["anc_bce"], st["bank_bce"] = float(la), float(lb)
        st["n_anc_ok"] = float((oka & torch.isfinite(ya).all(-1)).sum())
        st["n_bank_ok"] = float((okb & torch.isfinite(yb).all(-1)).sum())
    st["loss"] = float(loss.detach())
    return loss, st


def dummy_backward(out: Dict) -> None:
    """DDP: a skipped step (non-finite loss on some rank) still runs one zero-weighted backward on every rank."""
    terms = [out[k].float().nan_to_num(0.0, 0.0, 0.0).sum() * 0.0 for k in ("l_diff", "l_traj", "logit")
             if k in out and torch.is_tensor(out[k])]
    if terms:
        torch.stack(terms).sum().backward()


# ----------------------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def run_eval(net: FN.FPDRefinerNet, ds, a, dev, D, with_score: bool, collate_fn=None) -> Dict[str, np.ndarray]:
    """Sharded evaluation (rank r takes the r-th contiguous slice) -> arrays of all ranks.  Items need parents
    (bank.ParentItems / BankDataset(parents=True)); with_score: the 64-candidate sets (cand + bank_traj)."""
    net.eval()
    shard = np.array_split(np.arange(len(ds)), D.world)[D.rank]
    dl = T2.make_loader(torch.utils.data.Subset(ds, shard.tolist()), batch_size=a.eval_batch,
                        workers=a.workers if len(shard) else 0, pin=dev.type == "cuda", collate_fn=collate_fn)
    acc: Dict[str, list] = {k: [] for k in ("ade_par", "ade_draw", "shift", "prob", "y", "ok")}
    sums = np.zeros(3)
    bad, ntok = 0, 0
    if len(shard):
        for batch in dl:
            batch = TC.to_dev(batch, dev)
            okb = bev_ok_of(batch)
            bad += int((~okb).sum())
            ntok += int(okb.numel())
            status, gt, par = batch["status"].float(), batch["gt_traj"].float(), batch["parents"].float()
            ctx = net.scene(batch["bev"])
            keys = list(batch["tokens"])
            ld, lt, st = FN.dit_loss(net, par, gt[:, None], okb[:, None], ctx, status, keys, a.seed, VAL_NOISE_EPOCH, 0)
            n = float(st["fpd/n_pairs"])
            sums += np.array([float(ld) * n, float(lt) * n, n])
            noise = FN.token_noise(keys, (BK.N_PARENTS, BK.BANK_DRAWS, 8, 3), a.seed, BK.BANK_NOISE_EPOCH,
                                   BK.BANK_STREAM, device=dev)
            draws = net.sample_draws(par, ctx, status, noise, a.steps)               # [B, 16, 2, 8, 3]
            m = okb.cpu().numpy()
            acc["ade_par"].append(FN.mean_xy_dist(par, gt[:, None]).cpu().numpy()[m])
            acc["ade_draw"].append(FN.mean_xy_dist(draws, gt[:, None, None]).cpu().numpy()[m])
            acc["shift"].append(FN.mean_xy_dist(draws, par[:, :, None]).cpu().numpy()[m])
            if with_score:
                cset = torch.cat([batch["cand"].float(), batch["bank_traj"].float()], 1)
                acc["prob"].append(torch.sigmoid(net.score(cset, ctx, status).float()).cpu().numpy())
                acc["y"].append(torch.cat([batch["y"], batch["bank_y"]], 1).float().cpu().numpy())
                acc["ok"].append((torch.cat([batch["y_ok"], batch["bank_ok"]], 1).bool() & okb[:, None]).cpu().numpy())
    net.train()
    mine = {k: (np.concatenate(v) if v else None) for k, v in acc.items()}
    mine.update(_sums=sums, _bad=bad, _ntok=ntok)
    parts = D.all_gather(mine)
    R: Dict[str, np.ndarray] = {}
    for k in acc:
        arrs = [p[k] for p in parts if p[k] is not None]
        if arrs:
            R[k] = np.concatenate(arrs)
    R["sums"] = np.sum([p["_sums"] for p in parts], 0)
    R["n_bev_bad"] = np.asarray(sum(int(p["_bad"]) for p in parts))
    R["n_tok"] = np.asarray(sum(int(p["_ntok"]) for p in parts))
    return R


def eval_metrics(R: Dict[str, np.ndarray], score_keys, with_score: bool) -> Dict[str, float]:
    m: Dict[str, float] = {"n_tokens": int(R["n_tok"]), "n_bev_bad": int(R["n_bev_bad"])}
    s = R["sums"]
    m["n_pairs"] = int(s[2])
    if s[2] > 0:
        m["l_diff"], m["l_traj"] = float(s[0] / s[2]), float(s[1] / s[2])
    if "ade_par" in R and len(R["ade_par"]):
        P, Dr, Sh = R["ade_par"], R["ade_draw"], R["shift"]                     # [T, 16], [T, 16, 2], [T, 16, 2]
        t, k = np.arange(len(P)), P.argmin(1)
        dm = Dr[t, k].min(-1)
        m.update(parent_min_ade=float(P.min(1).mean()), bank_min_ade=float(Dr.reshape(len(Dr), -1).min(1).mean()),
                 match_ade=float(P[t, k].mean()), match_draw_min_ade=float(dm.mean()),
                 frac_match_improved=float((dm < P[t, k]).mean()), draw_shift=float(Sh.mean()))
    if with_score and "prob" in R and len(R["prob"]):
        Pr, Y, OK = R["prob"], R["y"], R["ok"]
        ka = Pr.shape[1] - BK.BANK_K
        m.update(T2.key_metrics(Pr.reshape(-1, 5), Y.reshape(-1, 5), OK.reshape(-1)))
        m.update(T2.key_metrics(Pr[:, :ka].reshape(-1, 5), Y[:, :ka].reshape(-1, 5), OK[:, :ka].reshape(-1), "anc_"))
        m.update(T2.key_metrics(Pr[:, ka:].reshape(-1, 5), Y[:, ka:].reshape(-1, 5), OK[:, ka:].reshape(-1), "bank_"))
        v = [m.get(f"auc_fail_{k}") for k in score_keys if k in T2.FAIL_KEYS]
        v = [float(x) for x in v if x is not None and math.isfinite(float(x))]
        m["auc_keys_mean"] = float(np.mean(v)) if v else float("nan")
    return m


# ----------------------------------------------------------------------------------------------- checkpoints
def _torch_save(obj, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _recipe(cfg: Dict) -> Dict:
    keys = ("epochs", "lr", "wd", "warmup_steps", "tokens_per_batch", "clip", "amp", "lambda_diff", "lambda_traj",
            "lambda_score", "score_keys", "world_size", "steps_total", "n_train", "seed", "sampler_seed",
            "noise_epoch_offset", "max_steps", "limit_tokens")
    return {k: cfg.get(k) for k in keys if k in cfg}


def teacher_config(a, rd: Path, net: FN.FPDRefinerNet, scfg: Dict) -> Dict:
    """config.json of the finished teacher (fpd_net.load_fpd_teacher reads trainer / arm / seed / steps / grid)."""
    dcfg = scfg.get("dit_cfg") or BK.read_json(rd / "dit" / "config.json", {}) or {}
    return dict(
        trainer=FN.FPD_TEACHER_TRAINER, arm=a.arm, teacher=TEACHER_OF_ARM[a.arm], score_keys=list(a.score_keys),
        ep_target=EP_TARGET, seed=int(a.seed), steps=int(a.steps), grid=int(a.grid), stage="scorer",
        split=a.split, val_split=a.val_split, init_from="none", c_in=256, bev_shape=scfg.get("bev_shape"),
        bev_input="teacher BEV f16 of CK2Dataset (TeacherCache / ResmapCache load_bev s_grid) = e2e kd_bev_0 (T) / "
                  "kd_bev_1 (M), fed as .float(), no z-score",
        sampler_seed=int(a.sampler_seed), n_parents=BK.N_PARENTS, n_bank_draws=BK.BANK_DRAWS, bank_k=BK.BANK_K,
        bank_stream=BK.BANK_STREAM, frozen_heads=[k for k in FN.CK_KEYS if k not in a.score_keys],
        recipe={"dit": _recipe(dcfg), "scorer": _recipe(scfg)},
        stages={"dit": {"dir": str(rd / "dit"), "ckpt_sha16": scfg.get("dit_ckpt_sha16"),
                        "done": BK.read_json(rd / "dit" / "done.json")},
                "bank": scfg.get("bank"),
                "scorer": {"dir": str(rd / "scorer"), "done": BK.read_json(rd / "scorer" / "done.json")}},
        init_report=scfg.get("init_report"), param_counts=net.param_counts(), git_head=U.git_head(),
        run_dir=str(rd), created=U.now())


def finalize_teacher(a, rd: Path, net: FN.FPDRefinerNet, scfg: Dict) -> Dict:
    """Write the teacher run dir: ckpt_last.pt (save_fpd_teacher), config.json, then done.json (last)."""
    cfg = teacher_config(a, rd, net, scfg)
    FN.save_fpd_teacher(rd, net, cfg)
    U.write_json(rd / "config.json", cfg)
    done = dict(run_dir=str(rd), arm=a.arm, teacher=TEACHER_OF_ARM[a.arm], trainer=FN.FPD_TEACHER_TRAINER,
                stage="scorer", ckpt="ckpt_last.pt", ckpt_sha16=U.sha256_file(rd / "ckpt_last.pt", 16),
                dit=BK.read_json(rd / "dit" / "done.json"), scorer=BK.read_json(rd / "scorer" / "done.json"),
                finished=U.now())
    U.write_json(rd / "done.json", done)
    print(f"FPD teacher {rd}: ckpt_last.pt sha16 {done['ckpt_sha16']}", flush=True)
    return done


def check_finished_stage_args(a, sd: Path) -> None:
    """The teacher files written for a finished stage scorer take arm / score keys / seed / DDIM steps / grid / split
    from the options (teacher_config): they must be the ones the stage ran with (its config.json), else SystemExit
    (called on every rank: the resume check of _train_stage is never reached for a finished stage)."""
    old = BK.read_json(sd / "config.json") or {}
    cur = vars(a)
    bad = [(k, old.get(k), cur.get(k)) for k in FINALIZE_KEYS if k in old and old.get(k) != cur.get(k)]
    if bad:
        raise SystemExit(f"{sd}: the finished stage ran with {bad} ((key, stage, now)): rerun with the stage's "
                         f"options to write the teacher files")


def finalize_from_stage(a, rd: Path) -> None:
    """stage scorer finished but the teacher files are missing (interrupted in between): write them from
    scorer/ckpt_last.pt (its cfg = the last session's, as at the normal end of the stage)."""
    sd = rd / "scorer"
    ck = torch.load(sd / "ckpt_last.pt", map_location="cpu", weights_only=False)
    net = build_net(a)
    net.load_state_dict(ck["model"], strict=True)
    finalize_teacher(a, rd, net, ck.get("cfg") or BK.read_json(sd / "config.json", {}) or {})


# ----------------------------------------------------------------------------------------------- train stages
def train_stage(a, stage: str, datasets: Optional[Dict] = None, collate_fn=None) -> Path:
    """Stage dit / scorer (resumable; 1 process or one per GPU under torchrun).  datasets: {split: CK2Dataset-like}
    injection for tests."""
    if stage not in ("dit", "scorer"):
        raise ValueError(stage)
    D = T2.Dist(a.device, a.dist_timeout_min)
    ok = False
    try:
        run = _train_stage(a, stage, D, datasets, collate_fn)
        ok = True
        return run
    finally:
        D.close(clean=ok)


def _stage_inputs(a, stage: str, rd: Path, datasets) -> Dict:
    """-> train / val datasets (+ dit checkpoint sha and bank summaries for stage scorer: each bank sampled from the
    finished dit checkpoint for exactly this run's dataset rows, labelled; bank.check_bank)."""
    out: Dict = {"dit_sha": None, "bank": {}}
    if stage == "scorer":
        dd = rd / "dit"
        if not (dd / "done.json").is_file():
            raise SystemExit(f"{dd}: stage dit not finished (no done.json)")
        out["dit_sha"] = U.sha256_file(dd / "ckpt_last.pt", 16)
    base_tr = get_base(a, a.split, datasets)
    base_va = get_base(a, a.val_split, datasets) if a.val_split != "none" else None
    if stage == "dit":
        out["train"] = base_tr
        out["val"] = BK.ParentItems(base_va) if base_va is not None else None
        return out
    for split, base in ((a.split, base_tr), (a.val_split, base_va)):
        if base is not None:
            out["bank"][split] = BK.check_bank(BK.bank_dir(rd, split), out["dit_sha"],
                                               rows_sha16=BK.sha16_rows(base.rows))
    out["train"] = BK.BankDataset(base_tr, BK.bank_dir(rd, a.split))
    out["val"] = BK.BankDataset(base_va, BK.bank_dir(rd, a.val_split), parents=True) if base_va is not None else None
    return out


def _train_stage(a, stage: str, D, datasets, collate_fn) -> Path:
    from torch.nn.parallel import DistributedDataParallel as DDP

    if datasets is None:
        require_bev64()                                    # before anything is written
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    torch.manual_seed(a.seed + D.rank)
    np.random.seed((a.seed + D.rank) % 2 ** 32)
    dev = D.device
    if a.tokens_per_batch % D.world:
        raise SystemExit(f"--tokens-per-batch {a.tokens_per_batch} not divisible by world size {D.world}")
    b_local = a.tokens_per_batch // D.world
    rd = Path(a.run_dir)
    sd = rd / stage
    if (rd / "done.json").is_file():
        if D.is0:
            print(f"{rd} is a finished FPD teacher (done.json): move it away to retrain", flush=True)
        return rd
    if D.is0:
        sd.mkdir(parents=True, exist_ok=True)
        if a.restart:
            for f in list(sd.glob("ckpt_*.pt")) + [sd / x for x in ("done.json", "train_log.jsonl", "val_metrics.jsonl",
                                                                     "config.json")]:
                f.unlink(missing_ok=True)
    D.barrier()
    if (sd / "done.json").is_file():
        if stage == "scorer":
            check_finished_stage_args(a, sd)
            if D.is0:
                finalize_from_stage(a, rd)
        D.barrier()
        if D.is0:
            print(f"{sd} finished (done.json); --restart to redo the stage", flush=True)
        return rd
    last = sd / "ckpt_last.pt"
    if last.is_file() and not a.resume:
        raise SystemExit(f"{last} exists: pass --resume to continue or --restart to start the stage over")
    inp = _stage_inputs(a, stage, rd, datasets)
    collate_fn = collate_fn or T2.collate()
    train_ds, val_ds = inp["train"], inp["val"]
    if len(train_ds) == 0:
        raise SystemExit(f"stage {stage}: no training token")
    net = build_net(a)
    init_rep: Dict = {"init_from": "none (construction, seed %d)" % a.seed}
    if stage == "scorer":
        dck = rd / "dit" / "ckpt_last.pt"
        ck = torch.load(dck, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"], strict=True)
        init_rep = {"init_from": str(dck), "init_ckpt_sha16": inp["dit_sha"]}
        inp["dit_cfg"] = _recipe(ck.get("cfg") or {})              # the dit stage's last session (its recipe)
        del ck
    frozen = freeze_for_stage(net, stage, a.score_keys)
    if stage == "scorer" and a.score_prior and D.is0:
        prior = BK.bank_label_prior(train_ds, a.prior_rows)
        if prior is not None:
            net.scorer.set_prior(prior)                    # DDP broadcasts rank 0's parameters
            init_rep["score_prior"] = [float(x) for x in prior]
            init_rep["score_prior_source"] = "bank_label_prior (epoch-0 anchors + bank, official targets)"
    net = net.to(dev)
    params = trainable(net)
    use_amp = bool(a.amp) and dev.type == "cuda"
    bev_shape = list(np.asarray(train_ds[0]["bev"]).shape) if D.is0 else None
    cfg = dict(vars(a), stage=stage, trainer=FN.FPD_TEACHER_TRAINER, teacher=TEACHER_OF_ARM[a.arm],
               git_head=U.git_head(), stage_dir=str(sd), device_resolved=str(dev),
               cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"), world_size=D.world, tokens_per_rank=b_local,
               global_batch=a.tokens_per_batch, n_train=len(train_ds), n_val=len(val_ds) if val_ds is not None else 0,
               train_data=train_ds.describe() if hasattr(train_ds, "describe") else None,
               val_data=(val_ds.describe() if hasattr(val_ds, "describe") else None) if val_ds is not None else None,
               init_report=init_rep, frozen=frozen, bev_shape=bev_shape, noise_epoch_offset=NOISE_EPOCH_OFFSET[stage],
               dit_ckpt_sha16=inp["dit_sha"], dit_cfg=inp.get("dit_cfg"), bank=inp["bank"] or None,
               bank_traj_sha16=(inp["bank"].get(a.split) or {}).get("traj_sha16") if inp["bank"] else None,
               param_counts=net.param_counts(), param_count_trainable=int(sum(p.numel() for p in params)),
               amp_resolved="fp16 autocast + GradScaler" if use_amp else "fp32")
    steps_per_epoch = len(TC.EpochBatches(len(train_ds), a.tokens_per_batch, a.seed, 0))
    if steps_per_epoch == 0:
        raise SystemExit(f"no full batch of {a.tokens_per_batch} in {len(train_ds)} training tokens")
    total = steps_per_epoch * a.epochs
    if a.max_steps:
        total = min(total, a.max_steps)
    cfg.update(steps_per_epoch=steps_per_epoch, steps_total=total)
    old = U.read_json(sd / "config.json")
    if old is not None and a.resume:
        for k in RESUME_KEYS[stage]:
            if old.get(k) != cfg.get(k):
                raise SystemExit(f"{sd}: resume with {k}={cfg.get(k)!r} but config.json has {old.get(k)!r}")
    if D.is0 and (old is None or a.restart):
        U.write_json(sd / "config.json", cfg)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, TC.lr_lambda(min(a.warmup_steps, max(1, total // 10))
                                                                if a.max_steps else a.warmup_steps, total))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    epoch, step, bstart, n_skip = 0, 0, 0, {"loss": 0, "grad": 0}
    pending = None
    if last.is_file():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        epoch, step, bstart = int(ck["epoch"]), int(ck["step"]), int(ck.get("batch_in_epoch", 0))
        n_skip = dict(ck.get("n_skip", n_skip))
        pending = ck.get("eval_pending")
        if bstart >= steps_per_epoch:
            epoch, bstart = epoch + 1, 0
        if D.is0:
            print(f"resumed {last}: epoch {epoch} batch {bstart} step {step}"
                  + (f" (eval of epoch {pending['epoch']} pending)" if pending else ""), flush=True)
    smod = StageModel(net, a.seed)
    model = (DDP(smod, device_ids=[dev.index] if dev.type == "cuda" else None, broadcast_buffers=True)
             if D.world > 1 else smod)
    logp, valp = sd / "train_log.jsonl", sd / "val_metrics.jsonl"
    with_score = stage == "scorer"

    def save(ep: int, bi: int, eval_pending=None) -> None:
        if D.is0:
            _torch_save(dict(model=net.state_dict(), cfg=cfg, epoch=ep, step=step, optim=opt.state_dict(),
                             sched=sched.state_dict(), scaler=scaler.state_dict(), batch_in_epoch=bi,
                             n_skip=dict(n_skip), eval_pending=eval_pending, world_size=D.world), last)

    def recorded(ep: int, st_: int) -> bool:
        if not valp.is_file():
            return False
        for line in valp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("epoch") == ep and r.get("step") == st_:
                    return True
        return False

    def finish_epoch(pend: Dict) -> None:
        """Evaluation of the weights now in ckpt_last.pt (saved with eval_pending = pend; skipped when already in
        val_metrics.jsonl), then ckpt_last.pt re-saved without the marker and, for a finished epoch, the weights
        copied to ckpt_ep<e>.pt."""
        do_eval = bool(pend.get("eval", True)) and val_ds is not None
        if do_eval and not D.bcast(recorded(pend["epoch"], pend["step"]) if D.is0 else None):
            R = run_eval(net, val_ds, a, dev, D, with_score, collate_fn)
            m = eval_metrics(R, a.score_keys, with_score)
            TC.check_bev_bad(int(m.get("n_bev_bad", 0)), int(m.get("n_tokens", 0)), f"eval {a.val_split}", force=True)
            if D.is0:
                U.log_jsonl(valp, dict(epoch=pend["epoch"], step=pend["step"], split=a.val_split, kind=stage,
                                       partial_epoch=bool(pend["partial"]), **m))
        save(pend["save_epoch"], pend["save_bi"])
        if D.is0 and not pend["partial"]:
            _torch_save(dict(model=net.state_dict(), cfg=cfg, epoch=pend["epoch"], step=pend["step"]),
                        sd / f"ckpt_ep{pend['epoch']}.pt")
        D.barrier()

    if pending:
        finish_epoch(pending)
    stop = step >= total
    t_run = time.time()
    bad_run, tok_run = 0, 0
    while epoch < a.epochs and not stop:
        model.train()
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        sampler = T2.DistEpochBatches(len(train_ds), a.tokens_per_batch, a.seed, epoch, bstart, D.rank, D.world)
        dl = T2.make_loader(train_ds, batch_sampler=sampler, workers=a.workers, pin=dev.type == "cuda",
                            collate_fn=collate_fn)
        t0, t_win, n_win, agg, nb = time.time(), time.time(), 0, {}, 0
        bi = bstart
        for batch in dl:
            batch = TC.to_dev(batch, dev)
            out = forward(model, batch, stage, epoch, step, use_amp, dev)
            loss, st = compute_loss(out, batch, a, stage)
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
            n_win += T
            bad_run += int(st.get("n_bev_bad", 0))
            tok_run += T
            for k, v in st.items():
                agg[k] = agg.get(k, 0.0) + v
            if step % a.log_every == 0 or step == total:
                g = merge_stats(D.all_gather(st))
                bad_all, tok_all, n_all = D.all_sum([bad_run, tok_run, n_win])
                dt = time.time() - t_win
                if D.is0:
                    rec = dict(kind="step", stage=stage, epoch=epoch, step=step, batch=bi, lr=sched.get_last_lr()[0],
                               tok_per_s=round(n_all / max(dt, 1e-6), 2), n_skip_loss=n_skip["loss"],
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
        tot: Dict[str, float] = {}
        for p in D.all_gather(agg):
            for k, v in p.items():
                tot[k] = tot.get(k, 0.0) + v
        nb_all = max(nb, 1)
        mean = {k: (v / nb_all if k in SUM_KEYS else v / (nb_all * D.world)) for k, v in tot.items()}
        bad_all, tok_all = D.all_sum([bad_run, tok_run])
        if D.is0:
            U.log_jsonl(logp, dict(kind="epoch", stage=stage, epoch=epoch, step=step, sec=round(time.time() - t0, 1),
                                   n_tok=int(nb * a.tokens_per_batch), world=D.world,
                                   train={k: round(v, 6) for k, v in mean.items()}))
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
    summ = dict(stage=stage, run_dir=str(rd), arm=a.arm, steps=step, epochs_done=epoch, epochs=a.epochs,
                total_steps=total, max_steps=a.max_steps, n_skip=n_skip, n_bev_bad_last_session=int(bad_all),
                n_tok_last_session=int(tok_all), world=D.world, sec=round(time.time() - t_run, 1),
                ckpt_sha16=U.sha256_file(last, 16) if (D.is0 and last.is_file()) else None, finished=U.now())
    if D.is0:
        U.write_json(sd / "done.json", summ)
        print(f"done {sd}: {summ}", flush=True)
        if stage == "scorer":
            finalize_teacher(a, rd, net, cfg)
    D.barrier()
    return rd


# ----------------------------------------------------------------------------------------------- bank stage
def sample_bank(a, datasets: Optional[Dict] = None, collate_fn=None) -> Path:
    """Stage bank (resumable per row; 1 process or one per GPU): DiT draws of every token of --bank-splits."""
    D = T2.Dist(a.device, a.dist_timeout_min)
    ok = False
    try:
        run = _sample_bank(a, D, datasets, collate_fn)
        ok = True
        return run
    finally:
        D.close(clean=ok)


def load_dit(a, rd: Path, dev) -> Tuple[FN.FPDRefinerNet, str]:
    dd = rd / "dit"
    if not (dd / "done.json").is_file():
        raise SystemExit(f"{dd}: stage dit not finished (no done.json)")
    dcfg = BK.read_json(dd / "config.json", {}) or {}
    for k, v in (("arm", a.arm), ("seed", a.seed), ("steps", a.steps), ("grid", a.grid)):
        if k in dcfg and dcfg[k] != v:
            raise SystemExit(f"{dd}: stage dit ran with {k}={dcfg[k]!r}, now {v!r}")
    p = dd / "ckpt_last.pt"
    sha = U.sha256_file(p, 16)
    net = build_net(a)
    net.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["model"], strict=True)
    net.requires_grad_(False)
    net.eval()
    return net.to(dev), sha


def _bank_files(bd: Path, which: str = "all") -> List[Path]:
    samp = [BK.F_TOKENS, BK.F_TRAJ, BK.F_PARENT, BK.F_SAMPLED, BK.F_SAMPLE_META, BK.F_SAMPLE_DONE]
    lab = [BK.F_LABELS, BK.F_OK, BK.F_LSTATE, BK.F_LPLAN, BK.F_LMETA, BK.F_ERRORS]
    return [bd / f for f in (samp + lab if which == "all" else lab)]


def _sample_bank(a, D, datasets, collate_fn) -> Path:
    if datasets is None:
        require_bev64()
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    dev = D.device
    rd = Path(a.run_dir)
    if (rd / "done.json").is_file():
        if D.is0:
            print(f"{rd} is a finished FPD teacher (done.json): move it away to resample", flush=True)
        return rd
    net, sha = load_dit(a, rd, dev)
    collate_fn = collate_fn or T2.collate()
    for split in a.bank_splits:
        bd = BK.bank_dir(rd, split)
        if a.restart and D.is0:
            for f in _bank_files(bd):
                f.unlink(missing_ok=True)
        D.barrier()
        base = get_base(a, split, datasets)
        sample_split(a, D, net, base, bd, split, sha, dev, collate_fn)
    return rd


def sample_split(a, D, net: FN.FPDRefinerNet, base, bd: Path, split: str, dit_sha: str, dev, collate_fn) -> Dict:
    """DiT bank of one split -> bd (bank.py layout).  Rows of the base dataset (sampler epoch 0) are sharded over the
    ranks; rows already sampled are skipped (BEV-unreadable ones are tried again); sample_done.json is written last by
    rank 0, once no row is left and at most train_ck.BEV_BAD_MAX of the split's rows had no readable BEV."""
    if hasattr(base, "set_epoch"):
        base.set_epoch(0)                                  # the bank parents are the epoch-0 draw
    rows = np.asarray(base.rows, np.int64)
    N = int(base.n)
    plan = dict(split=split, arm=a.arm, n_packed=N, n_rows=int(len(rows)), rows_sha16=BK.sha16_rows(rows),
                dit_ckpt_sha16=dit_sha, seed=int(a.seed), sampler_seed=int(a.sampler_seed), sampler_epoch=0,
                stream=BK.BANK_STREAM, noise_epoch=BK.BANK_NOISE_EPOCH, steps=int(a.steps), grid=int(a.grid),
                n_parents=BK.N_PARENTS, n_draws=BK.BANK_DRAWS, bank_k=BK.BANK_K,
                layout="traj [N, 32, 8, 3]: column 2 k + d = draw d of parent k; parents = near + mid anchors of the "
                       "sampler's epoch-0 draw in GT-rank order (parent_idx)",
                base=base.describe() if hasattr(base, "describe") else None, created=U.now())
    old = BK.read_json(bd / BK.F_SAMPLE_META)
    if old is not None:
        bad = [k for k in SAMPLE_PLAN_KEYS if old.get(k) != plan.get(k)]
        if bad:
            raise SystemExit(f"{bd}: sampled with {[(k, old.get(k), plan.get(k)) for k in bad]} (old, now): move it "
                             f"away (or --restart) to resample")
    done = BK.read_json(bd / BK.F_SAMPLE_DONE)
    if done is not None:
        if D.is0:
            print(f"{bd}: bank sampled ({done.get('n_sampled')} rows, traj sha16 {done.get('traj_sha16')})", flush=True)
        return done
    if D.is0:
        bd.mkdir(parents=True, exist_ok=True)
        if old is None:
            BK.token_table(base).to_parquet(bd / BK.F_TOKENS, index=False)
            U.write_json(bd / BK.F_SAMPLE_META, plan)
        U.open_memmap(bd / BK.F_TRAJ, (N, BK.BANK_K, 8, 3), np.float32, fill=np.nan)
        U.open_memmap(bd / BK.F_PARENT, (N, BK.N_PARENTS), np.int64, fill=-1)
        U.open_memmap(bd / BK.F_SAMPLED, (N,), np.uint8, fill=0)
    D.barrier()
    traj = U.open_memmap(bd / BK.F_TRAJ, (N, BK.BANK_K, 8, 3), np.float32)
    pidx = U.open_memmap(bd / BK.F_PARENT, (N, BK.N_PARENTS), np.int64)
    state = U.open_memmap(bd / BK.F_SAMPLED, (N,), np.uint8)
    # dataset positions not sampled yet, or whose BEV was unreadable (2: a rerun tries them again, e.g. after a cache
    # mount problem stopped the previous run)
    todo = np.flatnonzero(np.asarray(state[rows]) != 1)
    shard = np.array_split(todo, D.world)[D.rank]
    t0, n_done, n_bad, nb = time.time(), 0, 0, 0
    if len(shard):
        dl = T2.make_loader(torch.utils.data.Subset(BK.ParentItems(base), shard.tolist()), batch_size=a.sample_batch,
                            workers=a.workers, pin=dev.type == "cuda", collate_fn=collate_fn)
        with torch.no_grad():
            for batch in dl:
                r = batch["rows"].cpu().numpy()
                okb = batch["bev_ok"].bool().reshape(-1).cpu().numpy() if "bev_ok" in batch else np.ones(len(r), bool)
                ctx = net.scene(batch["bev"].to(dev))
                noise = FN.token_noise(list(batch["tokens"]), (BK.N_PARENTS, BK.BANK_DRAWS, 8, 3), a.seed,
                                       BK.BANK_NOISE_EPOCH, BK.BANK_STREAM, device=dev)
                draws = net.sample_draws(batch["parents"].to(dev).float(), ctx, batch["status"].to(dev).float(), noise,
                                         a.steps)
                out = draws.reshape(len(r), BK.BANK_K, 8, 3).float().cpu().numpy()
                if okb.any():
                    traj[r[okb]] = out[okb]
                pidx[r] = batch["parent_idx"].cpu().numpy()
                traj.flush()
                pidx.flush()
                state[r[okb]] = 1                         # written after the data (resume never sees half a row)
                state[r[~okb]] = 2
                state.flush()
                n_done += len(r)
                n_bad += int((~okb).sum())
                nb += 1
                if D.is0 and (nb % 200 == 0 or n_done == len(shard)):
                    print(f"[{time.strftime('%T')}] bank {split}: rank 0 {n_done}/{len(shard)} tokens "
                          f"({n_done / max(time.time() - t0, 1e-9):.1f} tok/s)", flush=True)
    D.barrier()
    # the whole split, not this session only (every rank reads the same state file after the barrier, so every rank
    # stops alike): no row left, and the BEV-unreadable rows (never labelled, left out of stage scorer) within
    # train_ck.BEV_BAD_MAX -- else no sample_done.json, a rerun tries those rows again
    st = np.asarray(np.load(bd / BK.F_SAMPLED, mmap_mode="r"))[rows]
    if (st == 0).any():
        raise SystemExit(f"{bd}: {int((st == 0).sum())} rows left unsampled")
    TC.check_bev_bad(int((st == 2).sum()), len(rows), f"bank {split}", force=True)
    out_done = None
    if D.is0:
        out_done = dict(plan, n_sampled=int((st == 1).sum()), n_bev_bad=int((st == 2).sum()),
                        traj_sha16=BK.sha16_file(bd / BK.F_TRAJ), sec_last_session=round(time.time() - t0, 1),
                        world_last_session=D.world, finished=U.now())
        U.write_json(bd / BK.F_SAMPLE_DONE, out_done)
        print(f"bank {bd}: {out_done['n_sampled']} rows sampled, {out_done['n_bev_bad']} without BEV, traj sha16 "
              f"{out_done['traj_sha16']} (rank 0 this session: {n_done} rows, {n_bad} without BEV)", flush=True)
    D.barrier()
    return out_done or {}


# ----------------------------------------------------------------------------------------------- main
def main(argv=None):
    a = resolve_args(get_parser().parse_args(argv))
    if a.stage == "bank":
        sample_bank(a)
    else:
        train_stage(a, a.stage)


if __name__ == "__main__":
    main()
