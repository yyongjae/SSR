"""Stage 1/2: train a planning readout h on a frozen, cached BEV.

The BEV comes from a cache (ReSMap teacher or PARA-SSR student, see
BevCache); nothing upstream of it is loaded.  Planning loss is PARA-SSR's own
``compute_plan_loss`` so every S_* is on the student's scale.

    # S_own: h1 on the teacher BEV
    python tools/readout/train_readout.py --bev-cache /data3/kyungmin/kd_teacher_resmap \
        --targets plan_targets_navtrain.npz --preset h1 --out runs/teacher_h1_s0
    # S_ego: same readout, BEV unread (the floor every score is read against)
    ... --no-bev --out runs/ego_h1_s0
    # control: shuffled trajectory labels (must fall to/below the floor)
    ... --shuffle-labels --out runs/teacher_h1_shuffled
    # S_student: same command on the student cache
    ... --bev-cache /data3/kyungmin/student_bev/navtrain --out runs/student_h1_s0
    # S_transfer + A: teacher-trained h frozen, only a 1x1 adapter learns
    ... --bev-cache <student cache> --init-readout runs/teacher_h1_s0/readout.pt --adapter-only

Open-loop validation uses a log-level hold-out of the training logs because
the teacher cache holds navtrain's train_logs only.  PDMS is measured
separately by eval_readout_pdms.py.
"""
import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import _env  # noqa: F401
from navsim.agents.para_ssr.para_ssr_loss import compute_plan_loss
from navsim.agents.para_ssr.readout.bev_cache import BevCache
from navsim.agents.para_ssr.readout.plan_targets import PlanTargetStore, load_log_split, log_holdout
from navsim.agents.para_ssr.readout.readout import (
    READOUT_PRESETS, build_readout, load_readout, readout_checkpoint,
)


class ReadoutData(Dataset):
    def __init__(self, tokens, store, cache, use_bev, label_perm=None):
        self.tokens, self.store, self.cache, self.use_bev = tokens, store, cache, use_bev
        self.label_perm = label_perm

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        tok = self.tokens[i]
        t = self.store.get(tok)
        item = {
            "command": torch.from_numpy(t["command"]),
            "ego": torch.from_numpy(t["ego"]),
        }
        lab = self.store.get(self.tokens[self.label_perm[i]]) if self.label_perm is not None else t
        item["offsets"] = torch.from_numpy(lab["offsets"])
        item["mask"] = torch.from_numpy(lab["mask"])
        if self.use_bev:
            item["bev"] = torch.from_numpy(self.cache.bev(tok))
        return item


class Adapter(nn.Module):
    """Design s05 alignment adapter: identity-initialised 1x1 conv."""

    def __init__(self, dims):
        super().__init__()
        self.conv = nn.Conv2d(dims, dims, 1)
        with torch.no_grad():
            self.conv.weight.copy_(torch.eye(dims).view(dims, dims, 1, 1))
            self.conv.bias.zero_()

    def forward(self, x):
        return self.conv(x)


def open_loop(pred_offsets, gt_offsets, mask):
    """L2 (m) at 1/2/3/4 s and ADE on x, y."""
    p = pred_offsets.cumsum(1)[..., :2]
    g = gt_offsets.cumsum(1)[..., :2]
    err = (p - g).norm(dim=-1) * mask
    out = {f"l2_{s}s": err[:, 2 * s - 1] for s in (1, 2, 3, 4)}
    out["ade"] = err.sum(1) / mask.sum(1).clamp(min=1)
    return out


def run_epoch(model, adapter, loader, device, opt=None, sched=None, heading_weight=0.5, amp=False):
    train = opt is not None
    model.train(train and adapter is None)
    if adapter is not None:
        adapter.train(train)
    sums, n = {}, 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        bev = batch.get("bev")
        if bev is not None:
            bev = bev.float()
            if adapter is not None:
                bev = adapter(bev)
        ctx = torch.autocast(device.type, dtype=torch.bfloat16) if amp else contextlib.nullcontext()
        with ctx:
            out = model(bev, batch["command"], batch["ego"])
        preds = out["ego_fut_preds"].float()
        loss, _ = compute_plan_loss(
            preds, batch["offsets"], batch["mask"], batch["command"], heading_weight=heading_weight
        )
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 10.0)
            opt.step()
            sched.step()
        bs = preds.shape[0]
        metrics = {"loss": loss.detach().expand(bs)}
        if not train:
            with torch.no_grad():
                metrics.update(open_loop(preds[:, 0], batch["offsets"], batch["mask"]))
                if bs > 1:
                    # How different z is across samples.  Near 0 means z is almost
                    # constant, and a cosine distillation on it has nothing to pull.
                    z = out["z"].float().flatten(1)
                    metrics["z_cos_spread"] = 1 - torch.nn.functional.cosine_similarity(z, z.roll(1, 0), dim=1)
                if bev is not None and bs > 1:
                    # Ego-shortcut check: the same batch with BEVs rotated by one
                    # sample.  If this matches l2_4s, the readout ignores the BEV.
                    shuf = model(bev.roll(1, 0), batch["command"], batch["ego"])["ego_fut_preds"].float()
                    metrics["l2_4s_bev_shuffled"] = open_loop(shuf[:, 0], batch["offsets"], batch["mask"])["l2_4s"]
        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + float(v.sum())
        n += bs
    return {k: v / max(n, 1) for k, v in sums.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bev-cache", required=True)
    ap.add_argument("--targets", required=True, help="build_plan_targets.py output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="h1", choices=sorted(READOUT_PRESETS))
    ap.add_argument("--num-queries", type=int, default=1)
    ap.add_argument("--ego-inject", default="late", choices=["late", "early", "none"])
    ap.add_argument("--cmd-inject", default="query", choices=["query", "late"])
    ap.add_argument("--no-bev", action="store_true", help="h_ego floor")
    ap.add_argument("--shuffle-labels", action="store_true", help="control: permuted trajectories")
    ap.add_argument("--init-readout", help="start from this readout checkpoint")
    ap.add_argument("--adapter-only", action="store_true", help="freeze --init-readout, train a 1x1 adapter")
    ap.add_argument("--holdout-frac", type=float, default=0.05, help="fraction of train logs for open-loop val")
    ap.add_argument("--max-train", type=int, default=0, help="smoke test: cap training samples")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (the loss stays fp32)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    store = PlanTargetStore(Path(args.targets))
    cache = BevCache(args.bev_cache)
    split = load_log_split("trainval")
    train_logs = set(split["train_logs"])
    # One token set for every arm (including --no-bev), so floors and scores
    # are measured on identical frames.
    rows = [(t, l) for t, l in zip(store.tokens, store.logs) if t in cache and l in train_logs]
    held = log_holdout([l for _, l in rows], args.holdout_frac, seed=0)
    train_tok = [t for t, l in rows if l not in held]
    val_tok = [t for t, l in rows if l in held]
    if args.max_train:
        rng = np.random.default_rng(args.seed)
        train_tok = sorted(rng.choice(train_tok, size=min(args.max_train, len(train_tok)), replace=False))
        val_tok = val_tok[: max(64, args.max_train // 10)]
    print(f"tokens: train {len(train_tok)}  val {len(val_tok)}  (held-out logs {len(held)})", flush=True)

    perm = np.random.default_rng(args.seed + 1).permutation(len(train_tok)) if args.shuffle_labels else None
    use_bev = not args.no_bev
    dl_kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                 persistent_workers=args.workers > 0)
    train_dl = DataLoader(ReadoutData(train_tok, store, cache, use_bev, perm), shuffle=True,
                          drop_last=True, **dl_kw)
    val_dl = DataLoader(ReadoutData(val_tok, store, cache, use_bev), shuffle=False, **dl_kw)

    device = torch.device(args.device)
    if args.init_readout:
        model = load_readout(args.init_readout)
        if model.cfg.use_bev != use_bev:
            raise SystemExit("--init-readout and --no-bev disagree")
    else:
        model = build_readout(
            args.preset, num_queries=args.num_queries, ego_inject=args.ego_inject,
            cmd_inject=args.cmd_inject, use_bev=use_bev,
        )
    model.to(device)
    adapter = None
    if args.adapter_only:
        if not args.init_readout or not use_bev:
            raise SystemExit("--adapter-only needs --init-readout and a BEV")
        model.eval().requires_grad_(False)
        adapter = Adapter(model.cfg.in_channels).to(device)
        params = list(adapter.parameters())
    else:
        unread = ("in_proj.", "pos_mlp.", "layers.", "query_pos.") if not use_bev else ()
        params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(unread)]

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * len(train_dl)
    warm = max(1, int(total * args.warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1)))
    )
    n_params = sum(p.numel() for p in params)
    meta = {"args": vars(args), "trainable_params": n_params, "readout_config": vars(model.cfg).copy(),
            "bev_cache_meta": cache.meta, "num_train": len(train_tok), "num_val": len(val_tok)}
    (out / "config.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"trainable params {n_params:,}", flush=True)

    history = []
    best = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        tr = run_epoch(model, adapter, train_dl, device, opt, sched, amp=args.amp)
        with torch.no_grad():
            va = run_epoch(model, adapter, val_dl, device, amp=args.amp)
        rec = {"epoch": epoch, "sec": round(time.time() - t0, 1),
               **{f"train/{k}": v for k, v in tr.items()}, **{f"val/{k}": v for k, v in va.items()}}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        extra = {"epoch": epoch, "val": va, "args": vars(args)}
        if adapter is not None:
            extra["adapter"] = adapter.state_dict()
        ckpt = readout_checkpoint(model, **extra)
        torch.save(ckpt, out / "readout_last.pt")
        if va["loss"] < best:
            best = va["loss"]
            torch.save(ckpt, out / "readout.pt")
    print(f"done; best val loss {best:.4f} -> {out / 'readout.pt'}")


if __name__ == "__main__":
    main()
