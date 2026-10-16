#!/usr/bin/env python
"""km 10/14: corridor head on cached BEV features -> GT corridor (tools/corridor/build_corridor.py targets).

  --src student : frozen CK2 bev_embed cache   <bevcache>/<split>/bev.npy [N, 5000, 256]  (50 x 100, row = forward)
  --src teacher : BEVFusion (camera+LiDAR) BEV <teacher>/samples/<tok[:2]>/<tok>.npz bev_feature [256, 50(x fwd), 100(y left)]
                  -> flipped to (forward, SSR x = right).  Upper bound of what distillation into the BEV could buy.
Targets (0.32 m, 100 x 200): static drivable SDF (m) and agent occupancy at t = 0.5 .. 4.0 s (8 planes).
Head: conv trunk on 50 x 100 -> x2 upsample -> 1 SDF plane (tanh * 8 m) + 8 occupancy logits.
Loss: smooth-L1 on the SDF, weighted toward the boundary (1 + 4 exp(-|sdf|)); BCE on occupancy (pos_weight 4).
Val (navtrain_val, every --eval-every epochs): SDF MAE in the |sdf| < 2 m band, drivable sign accuracy within 1 m
of the boundary, occupancy IoU at 0.5 / 2 / 4 s.  Training: navtrain_train only.

  CUDA_VISIBLE_DEVICES=0 $PY tools/corridor/train_corridor.py --src student --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CK = str(Path(__file__).resolve().parents[2])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

K = "/home/external-user/ssd/kyungmin_ck"
ROOT = f"{K}/km_runs/ck2e2e_bevkd/eval/root"
TEACHER = "/home/external-user/datasets/teacher_cache/bevfusion/cache_train_50x100/samples"
CLIP = 8.0


def log(msg, f=None):
    s = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(s, flush=True)
    if f:
        with open(f, "a") as fh:
            fh.write(s + "\n")


class DS(torch.utils.data.Dataset):
    def __init__(self, split, src, rows):
        import pandas as pd
        self.split, self.src, self.rows = split, src, np.asarray(rows)
        self.toks = pd.read_parquet(f"{ROOT}/packed/{split}/tokens.parquet").token.astype(str).to_numpy()
        self.cdir = f"{K}/corridor/{split}"
        self.bev = self.S = self.D = None

    def __len__(self):
        return len(self.rows)

    def _open(self):
        self.S = np.load(f"{self.cdir}/static_sdf.npy", mmap_mode="r")
        self.D = np.load(f"{self.cdir}/dyn_occ.npy", mmap_mode="r")
        if self.src == "student":
            self.bev = np.load(f"{K}/bevcache/{self.split}/bev.npy", mmap_mode="r")

    def __getitem__(self, i):
        if self.S is None:
            self._open()
        r = int(self.rows[i])
        if self.src == "student":
            x = np.asarray(self.bev[r], np.float16).T.reshape(256, 50, 100)
        else:
            t = self.toks[r]
            x = np.load(f"{TEACHER}/{t[:2]}/{t}.npz")["bev_feature"][:, :, ::-1]
        s = np.asarray(self.S[r], np.float32)
        d = np.unpackbits(np.asarray(self.D[r]), axis=-1).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(s), torch.from_numpy(d)


class Head(nn.Module):
    def __init__(self, cin=256, hid=192, t=8):
        super().__init__()
        def blk(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
        self.inp = nn.Sequential(nn.Conv2d(cin, hid, 1), nn.BatchNorm2d(hid), nn.ReLU(inplace=True))
        self.low = nn.Sequential(blk(hid, hid), blk(hid, hid), blk(hid, hid))
        self.up = nn.Sequential(nn.ConvTranspose2d(hid, hid // 2, 2, stride=2), nn.BatchNorm2d(hid // 2),
                                nn.ReLU(inplace=True), blk(hid // 2, hid // 2))
        self.out = nn.Conv2d(hid // 2, 1 + t, 1)

    def forward(self, x):
        y = self.out(self.up(self.low(self.inp(x.float()))))
        return CLIP * torch.tanh(y[:, 0] / CLIP), y[:, 1:]


def losses(sdf_p, occ_p, sdf, occ):
    w = 1.0 + 4.0 * torch.exp(-sdf.abs())
    ls = (w * F.smooth_l1_loss(sdf_p, sdf, beta=0.5, reduction="none")).mean()
    lo = F.binary_cross_entropy_with_logits(occ_p, occ, pos_weight=torch.tensor(4.0, device=occ.device))
    return ls, lo


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval()
    acc = {"mae2": [0.0, 0], "sign1": [0.0, 0], "iou": np.zeros((8, 2))}
    for x, s, d in dl:
        x, s, d = x.to(dev, non_blocking=True), s.to(dev), d.to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            sp, op = model(x)
        sp = sp.float()
        band = s.abs() < 2
        acc["mae2"][0] += (sp - s).abs()[band].sum().item(); acc["mae2"][1] += band.sum().item()
        b1 = s.abs() < 1
        acc["sign1"][0] += ((sp > 0) == (s > 0))[b1].sum().item(); acc["sign1"][1] += b1.sum().item()
        pb = op.float() > 0
        tb = d > 0.5
        acc["iou"][:, 0] += (pb & tb).sum((0, 2, 3)).cpu().numpy()
        acc["iou"][:, 1] += (pb | tb).sum((0, 2, 3)).cpu().numpy()
    model.train()
    iou = acc["iou"][:, 0] / np.maximum(acc["iou"][:, 1], 1)
    return {"sdf_mae_band2m": acc["mae2"][0] / max(acc["mae2"][1], 1),
            "sign_acc_band1m": acc["sign1"][0] / max(acc["sign1"][1], 1),
            "occ_iou_0.5s": float(iou[0]), "occ_iou_2s": float(iou[3]), "occ_iou_4s": float(iou[7])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", choices=["student", "teacher"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    L = out / "train.log"
    dev = torch.device("cuda:0")

    def rows(split):
        ok = np.load(f"{K}/corridor/{split}/ok.npy")
        if a.src == "student":
            dn = np.zeros_like(ok)
            for p in Path(f"{K}/bevcache/{split}").glob("done_*.npy"):
                dn |= np.load(p)
            ok &= dn
        else:
            import pandas as pd
            toks = pd.read_parquet(f"{ROOT}/packed/{split}/tokens.parquet").token.astype(str).to_numpy()
            ok &= np.array([Path(f"{TEACHER}/{t[:2]}/{t}.npz").exists() for t in toks])
        r = np.where(ok)[0]
        return r[: a.limit] if a.limit else r

    tr, va = rows("navtrain_train"), rows("navtrain_val")
    log(f"src {a.src}: train {len(tr)} val {len(va)}", L)
    dl = torch.utils.data.DataLoader(DS("navtrain_train", a.src, tr), batch_size=a.bs, shuffle=True,
                                     num_workers=a.workers, pin_memory=True, drop_last=True, persistent_workers=True)
    dv = torch.utils.data.DataLoader(DS("navtrain_val", a.src, va), batch_size=32, shuffle=False,
                                     num_workers=a.workers, pin_memory=True)
    model = Head().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(dl)
    warm = max(1, int(0.03 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, k / max(steps, 1)))))
    scaler = torch.cuda.amp.GradScaler()
    hist = []
    start = 0
    ck = out / "ckpt_last.pt"
    if ck.exists():
        st = torch.load(ck, map_location="cpu")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start, hist = st["epoch"], st["hist"]
        log(f"resumed at epoch {start}", L)
    for ep in range(start, a.epochs):
        t0, ns, so = time.time(), 0.0, 0.0
        for i, (x, s, d) in enumerate(dl):
            x, s, d = x.to(dev, non_blocking=True), s.to(dev, non_blocking=True), d.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                sp, op = model(x)
            ls, lo = losses(sp.float(), op.float(), s, d)
            loss = ls + lo
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            ns += ls.item(); so += lo.item()
            if i % 1000 == 0:
                log(f"ep {ep + 1} step {i}/{len(dl)} sdf {ls.item():.4f} occ {lo.item():.4f}", L)
        rec = {"epoch": ep + 1, "sdf": ns / len(dl), "occ": so / len(dl), "sec": round(time.time() - t0)}
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            rec.update(evaluate(model, dv, dev))
        hist.append(rec)
        log(json.dumps(rec), L)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep + 1, "hist": hist, "args": vars(a)}, ck)
        json.dump(hist, open(out / "hist.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
