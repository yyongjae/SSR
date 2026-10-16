#!/usr/bin/env python
"""km: train / evaluate the listwise selector (navsim/agents/para_ssr/ck/listwise.py) on the frozen CK2 pool dumps.

  cd ~/kyungmin/SSR-ck2km && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 \
    ~/miniconda3/envs/ssr/bin/python tools/ck/km/listwise_train.py --root <RUN>/eval/root --run <RUN name> \
    --train navtrain_train --val navtrain_val --out <dir> [--tau 0.1] [--cross 1] [--epochs 8] [--test navtest]

Inputs per split (eval_e2e2 dump with KM_DUMP_H=1, then its label stage):
  packed/<split>/{tokens.parquet, valid96, v2_final, v2_im, ok}
  infer/<run>/<split>/{score_logit, score_logit_lat, h, h_lat, e_lat, done}
  labels/<split>/lat96_<run>/{labels, ok} (+ pool96_<run> on val / test: eval_ck2.token_mask, same tokens as CK2's eval)
Target = official PDMS of the laterally corrected column (lat_mode 'on': the trajectory CK2 submits).
Every epoch is evaluated on --val (mean official PDMS of the selected column); the best val epoch is kept.
--test (navtest) is evaluated ONCE, with the kept checkpoint, only when given; its result is written next to val's.
"""
from __future__ import annotations

import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[3])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from navsim.agents.para_ssr.ck import listwise as LW  # noqa: E402
from navsim.agents.para_ssr.ck import select2 as S2  # noqa: E402
from navsim.agents.para_ssr.ck.constants import LBL  # noqa: E402
from tools.ck.e2e2 import eval_ck2 as EV  # noqa: E402

P_PDMS = LBL["pdms"]
BETA = 0.5          # CK2's navtrain_val choice (set 'all', lat 'on')


def log(msg: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


class Split:
    def __init__(self, root: Path, run: str, split: str, need_pool: bool):
        pk, inf = root / "packed" / split, root / "infer" / run / split
        self.name = split
        self.tokens = pd.read_parquet(pk / "tokens.parquet")
        n = len(self.tokens)
        ld = lambda d, k: np.load(d / f"{k}.npy", mmap_mode="r")
        self.valid96, self.v2_final, self.v2_im = ld(pk, "valid96"), ld(pk, "v2_final"), ld(pk, "v2_im")
        self.logit, self.logit_lat = ld(inf, "score_logit"), ld(inf, "score_logit_lat")
        self.h, self.h_lat, self.e_lat = ld(inf, "h"), ld(inf, "h_lat"), ld(inf, "e_lat")
        lat_d = root / "labels" / split / f"lat96_{run}"
        self.lab_lat, ok_lat = EV.load_labels(lat_d, n)
        if need_pool:
            self.lab_pool, ok_pool = EV.load_labels(root / "labels" / split / f"pool96_{run}", n)
        else:
            self.lab_pool, ok_pool = None, np.ones_like(ok_lat)
        m = EV.token_mask(ld(pk, "ok"), ld(inf, "done"), self.valid96, ok_pool, ok_lat)
        self.rows = np.flatnonzero(m)
        self.allowed = S2.allowed_mask(np.asarray(self.valid96), S2.VARIANT_SETS["all"])
        self.s0 = S2.pool_scores(np.asarray(self.v2_final), np.asarray(self.v2_im),
                                 1.0 / (1.0 + np.exp(-np.asarray(self.logit, np.float64))), BETA)
        log(f"{split}: {n} tokens, {len(self.rows)} evaluated (token mask{' with pool labels' if need_pool else ''})")

    def batch(self, rows: np.ndarray, dev) -> dict:
        rows = np.sort(rows)
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a[rows]), dtype=dt, device=dev)
        x = LW.pool_features(t(self.h), t(self.h_lat), t(self.logit), t(self.logit_lat), t(self.s0),
                             t(self.v2_final), t(self.v2_im), t(self.e_lat))
        return {"rows": rows, "x": x, "allowed": t(self.allowed, torch.bool), "s0": t(self.s0, torch.float64),
                "pdms": t(self.lab_lat[:, :, P_PDMS])}


@torch.no_grad()
def evaluate(head, S: Split, dev, bs: int = 512) -> dict:
    head.eval()
    idx = np.zeros(len(S.rows), np.int64)
    for i in range(0, len(S.rows), bs):
        b = S.batch(S.rows[i:i + bs], dev)
        r = (b["s0"] + head(b["x"], b["allowed"]).double()).cpu().numpy()
        idx[i:i + bs] = LW.select(r, b["allowed"].cpu().numpy())
    base = LW.select(S.s0[S.rows], S.allowed[S.rows])           # CK2 (beta 0.5, all, lat on) on the same tokens
    ar = np.arange(len(S.rows))
    lab = S.lab_lat[S.rows]
    out = {"n": int(len(S.rows)), "pdms": float(lab[ar, idx, P_PDMS].mean()),
           "pdms_ck2": float(lab[ar, base, P_PDMS].mean()), "frac_changed": float(np.mean(idx != base)),
           "oracle_lat": float(np.nanmax(np.where(S.allowed[S.rows], lab[..., P_PDMS], -1.0), 1).mean())}
    for k in ("nc", "dac", "ep", "ttc"):
        out[k] = float(lab[ar, idx, LBL[k]].mean())
        out[k + "_ck2"] = float(lab[ar, base, LBL[k]].mean())
    out["idx"] = idx
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--train", default="navtrain_train")
    ap.add_argument("--val", default="navtrain_val")
    ap.add_argument("--test", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--cross", type=int, default=1)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-only", action="store_true", help="no training: <out>/best.pt on --val (check) and --test")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    root = Path(a.root)
    if a.test_only:
        ck = torch.load(out / "best.pt", map_location=dev)
        t = ck["args"]
        head = LW.ListwiseHead(d=t["d"], n_layers=t["layers"], dropout=t["dropout"], cross=bool(t["cross"])).to(dev)
        head.load_state_dict(ck["model"])
        res = json.loads((out / "result.json").read_text())
        ev = evaluate(head, Split(root, a.run, a.val, need_pool=True), dev)
        assert abs(ev["pdms"] - res["val_best"]) < 1e-9, (ev["pdms"], res["val_best"])
        te = Split(root, a.run, a.test, need_pool=True)
        ev = evaluate(head, te, dev)
        np.save(out / f"idx_{a.test}.npy", ev["idx"])
        np.save(out / f"rows_{a.test}.npy", te.rows)
        res["test"] = {k: v for k, v in ev.items() if k != "idx"} | {"epoch": ck["epoch"]}
        (out / "result.json").write_text(json.dumps(res, indent=1))
        log(f"TEST {a.test} (epoch {ck['epoch']}): PDMS {ev['pdms']:.4f} (CK2 {ev['pdms_ck2']:.4f}), n {ev['n']}, "
            f"changed {ev['frac_changed']:.3f}")
        return 0
    tr = Split(root, a.run, a.train, need_pool=False)
    va = Split(root, a.run, a.val, need_pool=True)
    head = LW.ListwiseHead(d=a.d, n_layers=a.layers, dropout=a.dropout, cross=bool(a.cross)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (len(tr.rows) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    e0 = evaluate(head, va, dev)
    log(f"epoch 0 (head = 0): val PDMS {e0['pdms']:.4f} = CK2 {e0['pdms_ck2']:.4f}, changed {e0['frac_changed']:.4f}")
    assert e0["frac_changed"] == 0.0, "zero head must select CK2's columns"
    hist, best = [], (-1.0, -1)
    for ep in range(1, a.epochs + 1):
        head.train()
        perm = rng.permutation(tr.rows)
        t0, acc = time.time(), []
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            b = tr.batch(perm[i:i + a.bs], dev)
            r = b["s0"].float() + head(b["x"], b["allowed"])
            L = LW.listwise_loss(r, b["pdms"], b["allowed"], a.tau)
            opt.zero_grad(set_to_none=True)
            L["loss"].backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            acc.append(float(L["exp_pdms"]))
        ev = evaluate(head, va, dev)
        hist.append({k: v for k, v in ev.items() if k != "idx"} | {"epoch": ep, "train_exp_pdms": float(np.mean(acc)),
                                                                   "sec": round(time.time() - t0, 1)})
        log(f"epoch {ep}: train E[PDMS] {np.mean(acc):.4f} | val PDMS {ev['pdms']:.4f} (CK2 {ev['pdms_ck2']:.4f}, "
            f"changed {ev['frac_changed']:.3f}) NC {ev['nc']:.4f} DAC {ev['dac']:.4f} EP {ev['ep']:.4f} "
            f"TTC {ev['ttc']:.4f} | {time.time() - t0:.0f}s")
        if ev["pdms"] > best[0]:
            best = (ev["pdms"], ep)
            torch.save({"model": head.state_dict(), "args": vars(a), "epoch": ep}, out / "best.pt")
            np.save(out / f"idx_{a.val}.npy", ev["idx"])
    res = {"args": vars(a), "best_epoch": best[1], "val_best": best[0], "val_epoch0": e0["pdms"], "hist": hist}
    if a.test:
        head.load_state_dict(torch.load(out / "best.pt", map_location=dev)["model"])
        te = Split(root, a.run, a.test, need_pool=True)
        ev = evaluate(head, te, dev)
        np.save(out / f"idx_{a.test}.npy", ev["idx"])
        np.save(out / f"rows_{a.test}.npy", te.rows)
        res["test"] = {k: v for k, v in ev.items() if k != "idx"}
        log(f"TEST {a.test} (epoch {best[1]}): PDMS {ev['pdms']:.4f} (CK2 {ev['pdms_ck2']:.4f}), n {ev['n']}")
    (out / "result.json").write_text(json.dumps(res, indent=1))
    log(f"best val epoch {best[1]}: {best[0]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
