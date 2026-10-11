#!/usr/bin/env python
"""refiner-v3 forecaster PRE-TRAINING stage (review fix D3; protocol finding 5).  Forecast loss only.

  cd /workspace/yongjae/SSR-v3 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/yongjae/SSR-v3 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 nice -n 10 /venv/ssr/bin/python -B tools/ck/v3/pretrain_forecaster.py \
      --out D3/train/v3fc_pre_learned6 [--epochs 8] [--batch 64] [--lr 5e-4] [--resume]

Why: arm (a) of Step 1 used to train its learned forecaster from scratch together with the critic, so (a)'s critic
learned on raster inputs that kept changing while (b)'s CA forecaster was fixed from step 0.  Now the forecaster is
pre-trained here (own output dir, train split only) and arm (a) trains its critic with it FROZEN
(train_ck2 --t-fc-init <out>/forecaster.pt --t-fc-train frozen, the default): (a) and (b) both use a fixed
forecaster fitted on the augmented navtrain_train stream (CA: fit_ca_scales.py on the same augmentation).

What it trains: ck.temporal.forecast.LearnedForecaster(n_modes, seed) -- the exact module arm (a) builds (same init:
LearnedForecaster(n_modes, seed=<trainer seed>) under FORECASTER_SEED_OFFSET), on the object pack rows of
navtrain_train (D3/objpack_k96/bevfusion/navtrain_train, targets = matched GT futures, clean-row matching), with the
SAME training augmentation as the critic run (obj_pack.augment_objects: p_frame_drop / p_obj_drop / v_noise of
TemporalCfg) on its own seed (--aug-seed, default trainer seed + AUG_SEED_OFFSET: other noise realisations than the
critic run sees).  Inputs go through the same guards and trim as TemporalBranch.scene (check_causal, trim_objects).
Loss = temporal.loss.forecast_loss (WTA Laplace NLL + mode CE, FC_OBS steps only), AdamW, linear warm-up + cosine,
grad clip 1.0, fp32.  No best-epoch selection: forecaster.pt = the weights after the last epoch.

Evaluation after every epoch (clean inputs, no augmentation; metrics.ForecastMetrics of the learned forecaster AND of
the analytic CA reference with D3/ca_scales.json): navtrain_val (BEVFusion in-sample), navtest (BEVFusion
out-of-sample, GT from D3/gt_objects/navtest; review causality finding 2), and a fixed log-stratified subset of
navtrain_train (in-sample scenes).  overfit_index = (minADE_learned / minADE_CA)_val / (..)_train_subset; > 1.10 is
flagged (the critic of arm a would see better-than-deployable forecasts during training).
Outputs (--out): config.json (args, aug, provenance: code / pack / ca_scales shas), train_log.jsonl,
val_metrics.jsonl, ckpt_last.pt (resume), forecaster.pt ({'forecaster': state_dict, 'n_modes', 'seed', ...}),
done.json.  GPU: only GPU 0 or 1 (CUDA_VISIBLE_DEVICES exactly '0' or '1'); CPU for tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[3])
if sys.path[0] != REPO:
    sys.path.insert(0, REPO)
import navsim  # noqa: E402

assert navsim.__file__.startswith(REPO), navsim.__file__

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from typing import Dict, List, Optional  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from navsim.agents.para_ssr.ck.temporal import config as TCF  # noqa: E402

D3 = TCF.D3
AUG_SEED_OFFSET = 7919
VERSION = "v3_fc_pretrain_1"
OVERFIT_FLAG = 1.10
KEYS = ("obj_in", "obj_mask", "frame_valid", "fc_xyh", "fc_state")


def kst() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S KST", time.gmtime(time.time() + 9 * 3600))


# ----------------------------------------------------------------------------------------------- data
class PackRows(torch.utils.data.Dataset):
    """Object-pack rows (+ targets) of one split; augmentation (training) on (token, epoch, aug_seed)."""

    def __init__(self, split: str, cfg, rows: Optional[np.ndarray] = None, aug: bool = False, aug_seed: int = 0):
        from tools.ck.data.obj_pack import ObjPack
        self.pack = ObjPack(split, source=cfg.source, root=cfg.pack_root, targets=True)
        self.rows = np.arange(self.pack.n) if rows is None else np.asarray(rows, np.int64)
        ok = np.asarray(np.load(self.pack.dir / "obj_ok.npy", mmap_mode="r"), bool)
        self.rows = self.rows[ok[self.rows]]
        self.cfg, self.aug, self.aug_seed, self.epoch = cfg, bool(aug), int(aug_seed), 0
        self.split = split

    def set_epoch(self, e: int) -> None:
        self.epoch = int(e)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        from tools.ck.data.obj_pack import augment_objects
        r = int(self.rows[i])
        it = self.pack.item(r)
        if self.aug:
            it = augment_objects(it, self.pack.token(r), self.epoch, self.aug_seed, self.cfg)
        return {k: it[k] for k in KEYS}


def collate(items: List[Dict]) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.stack([it[k] for it in items])) for k in KEYS}


def stratified_rows(split: str, n: int, seed: int = 0) -> np.ndarray:
    from tools.ck import ckutil as U
    from tools.ck.data.obj_pack import packed_tokens  # noqa: F401  (packed order is the pack order)
    import pandas as pd
    tdf = pd.read_parquet(Path(TCF.D3).parents[0] / "packed" / split / "tokens.parquet")
    rows = np.arange(len(tdf))
    if n >= len(rows):
        return rows
    return np.sort(rows[U.log_stratified_rows(tdf["log"].values, n, seed=seed)])


# ----------------------------------------------------------------------------------------------- model step
def forward_fc(model, b: Dict[str, torch.Tensor], dev):
    from navsim.agents.para_ssr.ck.temporal.branch import check_causal
    from navsim.agents.para_ssr.ck.temporal.forecast import trim_objects
    oi = b["obj_in"].to(dev).float()
    om = b["obj_mask"].to(dev).bool()
    fv = b["frame_valid"].to(dev).bool()
    check_causal(oi, om, fv)
    oi, om, fv = trim_objects(oi, om, fv)
    return model(oi, om, fv), (oi, om, fv)


@torch.no_grad()
def evaluate(model, ds, dev, batch: int, workers: int, ca_b: Optional[torch.Tensor]) -> Dict:
    from navsim.agents.para_ssr.ck.temporal.forecast import ca_forecast
    from navsim.agents.para_ssr.ck.temporal.metrics import ForecastMetrics
    model.eval()
    m_l, m_c = ForecastMetrics(), ForecastMetrics()
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=collate)
    for b in dl:
        fc, (oi, om, fv) = forward_fc(model, b, dev)
        tgt = (b["fc_xyh"].to(dev), b["fc_state"].to(dev), b["obj_mask"][:, 0].to(dev))
        m_l.update(fc, *tgt)
        if ca_b is not None:
            m_c.update(ca_forecast(oi, om, fv, ca_b.to(dev)), *tgt)
    model.train()
    out = {"learned": m_l.result("fc_"), "n_rows": len(ds)}
    if ca_b is not None:
        out["ca"] = m_c.result("fc_")
        la, ca = out["learned"].get("fc_minade"), out["ca"].get("fc_minade")
        out["ade_ratio_learned_over_ca"] = (float(la) / float(ca)) if la and ca else None
    return out


def get_parser():
    d = TCF.TemporalCfg()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output dir (under D3/train or D3/smoke, or /tmp for tests)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0, help="= the critic run's --seed (forecaster init)")
    ap.add_argument("--aug-seed", type=int, default=None, help=f"default seed + {AUG_SEED_OFFSET}")
    ap.add_argument("--n-modes", type=int, default=d.n_modes)
    ap.add_argument("--frame-drop", type=float, default=d.p_frame_drop)
    ap.add_argument("--obj-drop", type=float, default=d.p_obj_drop)
    ap.add_argument("--vnoise", type=float, default=d.v_noise)
    ap.add_argument("--pack-root", default=d.pack_root)
    ap.add_argument("--ca-scales", default=d.ca_scales, help="CA reference metrics only (never trained on)")
    ap.add_argument("--train-split", default="navtrain_train")
    ap.add_argument("--eval-splits", default="navtrain_val,navtest")
    ap.add_argument("--train-eval-rows", type=int, default=4096, help="in-sample subset of the train split")
    ap.add_argument("--val-limit", type=int, default=0, help="first N rows of every eval split (smoke)")
    ap.add_argument("--limit-tokens", type=int, default=0, help="first N train rows (smoke)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (smoke)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    return ap


def build_cfg(a):
    t = TCF.TemporalCfg(on=True, forecaster="learned", n_modes=int(a.n_modes), p_frame_drop=float(a.frame_drop),
                        p_obj_drop=float(a.obj_drop), v_noise=float(a.vnoise), pack_root=str(a.pack_root),
                        ca_scales=str(a.ca_scales), fc_train="joint")
    t.validate("T")
    return t


def main(argv=None):
    from navsim.agents.para_ssr.ck.temporal.forecast import LearnedForecaster
    from navsim.agents.para_ssr.ck.temporal.loss import forecast_loss
    from tools.ck import ckutil as U
    from tools.ck import train_ck as TCK
    from tools.ck.v3 import provenance as PV

    a = get_parser().parse_args(argv)
    if a.workers > 16:
        raise SystemExit("--workers <= 16")
    out = Path(a.out)
    ro = str(out.resolve())
    if not (ro.startswith(str(D3.resolve()) + "/") or ro.startswith("/tmp/")):
        raise SystemExit(f"--out must be under {D3} (or /tmp for tests), got {out}")
    if a.device.startswith("cuda"):
        cv = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cv not in ("0", "1"):
            raise SystemExit(f"cuda only on GPU 0 or 1 (CUDA_VISIBLE_DEVICES='0' or '1'), got {cv!r}")
    dev = torch.device("cuda:0" if a.device.startswith("cuda") else "cpu")
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    cfg = build_cfg(a)
    aug_seed = int(a.seed) + AUG_SEED_OFFSET if a.aug_seed is None else int(a.aug_seed)
    rows = np.arange(a.limit_tokens) if a.limit_tokens else None
    train = PackRows(a.train_split, cfg, rows=rows, aug=True, aug_seed=aug_seed)
    evals = []
    for s in [x for x in a.eval_splits.split(",") if x]:
        r = np.arange(a.val_limit) if a.val_limit else None
        evals.append((s, PackRows(s, cfg, rows=r)))
    if a.train_eval_rows:
        tr = stratified_rows(a.train_split, a.train_eval_rows if not a.val_limit else min(a.val_limit,
                                                                                          a.train_eval_rows))
        evals.append((f"{a.train_split}_insample", PackRows(a.train_split, cfg, rows=tr)))
    ca_b = None
    if Path(a.ca_scales).is_file():
        from navsim.agents.para_ssr.ck.temporal.branch import load_ca_b
        ca_b = load_ca_b(a.ca_scales)
    model = LearnedForecaster(n_modes=cfg.n_modes, seed=int(a.seed)).to(dev)
    steps_per_epoch = math.ceil(len(train) / a.batch)
    total = steps_per_epoch * a.epochs
    if a.max_steps:
        total = min(total, a.max_steps)
    warm = min(a.warmup_steps, max(1, total // 10)) if a.max_steps else a.warmup_steps
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, TCK.lr_lambda(warm, total))
    out.mkdir(parents=True, exist_ok=True)
    conf = dict(version=VERSION, args=vars(a), n_modes=cfg.n_modes, seed=int(a.seed), epochs=int(a.epochs),
                aug=dict(p_frame_drop=cfg.p_frame_drop, p_obj_drop=cfg.p_obj_drop, v_noise=cfg.v_noise,
                         aug_seed=aug_seed, stream="objaug"),
                temporal=cfg.to_dict(), n_train=len(train), steps_per_epoch=steps_per_epoch, total_steps=total,
                param_count=int(sum(p.numel() for p in model.parameters())),
                provenance=PV.run_provenance(cfg, splits=[a.train_split] + [s for s, _ in evals if not
                                                                            s.endswith("_insample")]),
                device=str(dev), cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"), created=kst())
    last = out / "ckpt_last.pt"
    epoch, step = 0, 0
    if last.is_file():
        if not a.resume:
            raise SystemExit(f"{last} exists: pass --resume (or use a new --out)")
        old = U.read_json(out / "config.json") or {}
        for k in ("n_modes", "seed", "epochs", "aug"):
            if old.get(k) != conf[k]:
                raise SystemExit(f"resume with {k}={conf[k]!r} but config.json has {old.get(k)!r}")
        if ((old.get("provenance") or {}).get("data") or {}).get("packs") != conf["provenance"]["data"]["packs"]:
            raise SystemExit("resume with other object packs than config.json records")
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["forecaster"])
        opt.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        epoch, step = int(ck["epoch"]), int(ck["step"])
        print(f"resumed {last}: epoch {epoch} step {step}", flush=True)
    else:
        U.write_json(out / "config.json", conf)
    logp, valp = out / "train_log.jsonl", out / "val_metrics.jsonl"
    t_run = time.time()
    stop = step >= total
    while epoch < a.epochs and not stop:
        train.set_epoch(epoch)
        g = torch.Generator()
        g.manual_seed(int(a.seed) * 1000 + epoch)
        sampler = torch.utils.data.RandomSampler(train, generator=g)
        dl = torch.utils.data.DataLoader(train, batch_size=a.batch, sampler=sampler, num_workers=a.workers,
                                         collate_fn=collate, drop_last=False, pin_memory=dev.type == "cuda",
                                         persistent_workers=False)
        model.train()
        t0, agg, nb = time.time(), {}, 0
        skip_to = step - epoch * steps_per_epoch            # resumed mid-epoch: skip the batches already done
        for bi, b in enumerate(dl):
            if bi < skip_to:
                continue
            fc, _ = forward_fc(model, b, dev)
            loss, st = forecast_loss(fc, b["fc_xyh"].to(dev), b["fc_state"].to(dev), b["obj_mask"][:, 0].to(dev))
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                st["skipped"] = 1.0
            else:
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip if a.clip > 0 else float("inf"))
                st["grad_norm"] = float(gn)
                if math.isfinite(float(gn)):
                    opt.step()
                else:
                    st["skipped"] = 1.0
            sched.step()
            step += 1
            nb += 1
            for k, v in st.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            if step % a.log_every == 0 or step == total:
                rec = dict(kind="step", epoch=epoch, step=step, lr=sched.get_last_lr()[0],
                           sec=round(time.time() - t0, 1), **{k: round(v, 6) for k, v in st.items()})
                if dev.type == "cuda":
                    rec["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
                U.log_jsonl(logp, rec)
            if step >= total:
                stop = True
                break
        U.log_jsonl(logp, dict(kind="epoch", epoch=epoch, step=step, sec=round(time.time() - t0, 1),
                               train={k: round(v / max(nb, 1), 6) for k, v in agg.items()}))
        finished = (step - epoch * steps_per_epoch) >= steps_per_epoch
        res = {}
        for name, ds in evals:
            m = evaluate(model, ds, dev, max(a.batch, 64), a.workers, ca_b)
            res[name] = m
            U.log_jsonl(valp, dict(epoch=epoch, step=step, split=name, **m))
        rv = (res.get("navtrain_val") or {}).get("ade_ratio_learned_over_ca")
        rt = (res.get(f"{a.train_split}_insample") or {}).get("ade_ratio_learned_over_ca")
        if rv and rt:
            U.log_jsonl(valp, dict(epoch=epoch, step=step, split="overfit_index", value=rv / rt,
                                   flag=bool(rv / rt > OVERFIT_FLAG)))
        epoch_next = epoch + 1 if finished else epoch
        torch.save(dict(forecaster=model.state_dict(), optim=opt.state_dict(), sched=sched.state_dict(),
                        epoch=epoch_next, step=step, n_modes=cfg.n_modes, seed=int(a.seed)), str(last) + ".tmp")
        os.replace(str(last) + ".tmp", last)
        epoch = epoch_next
    fpath = out / "forecaster.pt"
    torch.save(dict(forecaster=model.state_dict(), n_modes=cfg.n_modes, seed=int(a.seed), version=VERSION,
                    step=step, epochs_done=epoch, config_sha256=PV.sha256_file(out / "config.json")),
               str(fpath) + ".tmp")
    os.replace(str(fpath) + ".tmp", fpath)
    U.write_json(out / "done.json", dict(steps=step, epochs_done=epoch, total_steps=total,
                                         sec=round(time.time() - t_run, 1), finished=kst(),
                                         forecaster_sha256=PV.sha256_file(fpath)))
    print(f"done {out}: forecaster.pt sha {PV.sha256_file(fpath)[:16]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
