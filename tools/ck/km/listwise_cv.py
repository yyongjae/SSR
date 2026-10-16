#!/usr/bin/env python
"""km: listwise selector trained on HELD-OUT data only (navtrain_val, K-fold by log), not on navtrain_train.

Why: v2 and the CK critic were trained on navtrain_train, so their features there are in-sample (CK2 selection 96.6
PDMS on train vs 91.4 on val, v2 top-1 94.3 vs 89.1). A selector stacked on those features learns to trust them
(listwise_train.py: train E[PDMS] 0.98, val within +-0.3 of CK2). Standard stacking: fit the meta-model on data the
base models never saw.

  K folds of navtrain_val grouped by log; a head per fold is trained on the other folds for a FIXED number of epochs
  (no early stopping on the held-out fold) and selects on its fold -> out-of-fold (OOF) PDMS over all val tokens,
  compared with CK2 on the same tokens (paired log bootstrap CI).  Configs are compared by OOF PDMS only.
  --test-only: the K fold heads (mean of their deltas) select on --test ONCE.

  cd ~/kyungmin/SSR-ck2km && source km_env.sh && CUDA_VISIBLE_DEVICES=1 $PY tools/ck/km/listwise_cv.py \
    --root <RUN>/eval/root --run <name> --out <dir> [--tau 0.1 --cross 1 --epochs 4 --init <best.pt>]
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
import torch  # noqa: E402

from navsim.agents.para_ssr.ck import listwise as LW  # noqa: E402
from navsim.agents.para_ssr.ck.constants import LBL  # noqa: E402
from tools.ck.km.listwise_train import P_PDMS, Split, log  # noqa: E402


def fold_of_rows(S: Split, k: int, seed: int) -> np.ndarray:
    """Fold id per evaluated row, grouped by log (logs shuffled with a fixed seed, dealt round robin)."""
    logs = np.asarray(S.tokens["log"].astype(str))[S.rows]
    u = np.unique(logs)
    np.random.default_rng(seed).shuffle(u)
    f = {lg: i % k for i, lg in enumerate(u)}
    return np.array([f[lg] for lg in logs])


@torch.no_grad()
def deltas(heads, S: Split, rows: np.ndarray, dev, bs: int = 512) -> np.ndarray:
    out = np.zeros((len(rows), S.allowed.shape[1]))
    for i in range(0, len(rows), bs):
        b = S.batch(rows[i:i + bs], dev)
        out[i:i + bs] = np.mean([h.eval()(b["x"], b["allowed"]).double().cpu().numpy() for h in heads], 0)
    return out


def score(S: Split, rows: np.ndarray, idx: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    ar = np.arange(len(rows))
    lab = S.lab_lat[rows]
    base = LW.select(S.s0[rows], S.allowed[rows])
    p, p0 = lab[ar, idx, P_PDMS], lab[ar, base, P_PDMS]
    logs = np.asarray(S.tokens["log"].astype(str))[rows]
    u, inv = np.unique(logs, return_inverse=True)
    d_sum, n_sum = np.bincount(inv, p - p0), np.bincount(inv)
    rng = np.random.default_rng(seed)
    bs = [(d_sum[s].sum() / n_sum[s].sum()) for s in (rng.integers(0, len(u), len(u)) for _ in range(n_boot))]
    out = {"n": int(len(rows)), "pdms": float(p.mean()), "pdms_ck2": float(p0.mean()), "delta": float((p - p0).mean()),
           "delta_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
           "frac_changed": float(np.mean(idx != base))}
    for k in ("nc", "dac", "ep", "ttc"):
        out[k], out[k + "_ck2"] = float(lab[ar, idx, LBL[k]].mean()), float(lab[ar, base, LBL[k]].mean())
    return out


def make_head(a, dev):
    return LW.ListwiseHead(d=a.d, n_layers=a.layers, dropout=a.dropout, cross=bool(a.cross)).to(dev)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--val", default="navtrain_val")
    ap.add_argument("--test", default="navtest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--cross", type=int, default=1)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--init", default="", help="start every fold head from this listwise_train.py best.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-only", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = Path(a.root)
    if a.test_only:
        res = json.loads((out / "result.json").read_text())
        t = argparse.Namespace(**res["args"])
        heads = []
        for f in range(t.k):
            h = make_head(t, dev)
            h.load_state_dict(torch.load(out / f"fold{f}.pt", map_location=dev))
            heads.append(h)
        te = Split(root, a.run, a.test, need_pool=True)
        d = deltas(heads, te, te.rows, dev)
        idx = LW.select(te.s0[te.rows] + d, te.allowed[te.rows])
        ev = score(te, te.rows, idx)
        np.save(out / f"idx_{a.test}.npy", idx)
        np.save(out / f"rows_{a.test}.npy", te.rows)
        res["test"] = ev
        (out / "result.json").write_text(json.dumps(res, indent=1))
        log(f"TEST {a.test} ({t.k} fold heads, mean delta): PDMS {ev['pdms']:.4f} (CK2 {ev['pdms_ck2']:.4f}, "
            f"d {100 * ev['delta']:+.2f} CI {100 * ev['delta_ci95'][0]:+.2f} {100 * ev['delta_ci95'][1]:+.2f}) "
            f"n {ev['n']} changed {ev['frac_changed']:.3f}")
        return 0
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    va = Split(root, a.run, a.val, need_pool=True)
    fold = fold_of_rows(va, a.k, a.seed)
    oof = np.zeros(len(va.rows), np.int64)
    folds = []
    for f in range(a.k):
        tr_rows, ho = va.rows[fold != f], np.flatnonzero(fold == f)
        head = make_head(a, dev)
        if a.init:
            head.load_state_dict(torch.load(a.init, map_location=dev)["model"])
        opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.wd)
        steps = a.epochs * (len(tr_rows) // a.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
        t0, acc = time.time(), []
        for ep in range(a.epochs):
            head.train()
            perm = rng.permutation(tr_rows)
            for i in range(0, len(perm) - a.bs + 1, a.bs):
                b = va.batch(perm[i:i + a.bs], dev)
                L = LW.listwise_loss(b["s0"].float() + head(b["x"], b["allowed"]), b["pdms"], b["allowed"], a.tau)
                opt.zero_grad(set_to_none=True)
                L["loss"].backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                opt.step()
                sched.step()
                acc.append(float(L["exp_pdms"]))
        torch.save(head.state_dict(), out / f"fold{f}.pt")
        r = va.s0[va.rows[ho]] + deltas([head], va, va.rows[ho], dev)
        oof[ho] = LW.select(r, va.allowed[va.rows[ho]])
        ev = score(va, va.rows[ho], oof[ho])
        folds.append(ev | {"fold": f, "train_exp_pdms_last_epoch": float(np.mean(acc[-(len(tr_rows) // a.bs):]))})
        log(f"fold {f}: n {ev['n']} PDMS {ev['pdms']:.4f} (CK2 {ev['pdms_ck2']:.4f}, d {100 * ev['delta']:+.2f}) "
            f"train E[PDMS] {folds[-1]['train_exp_pdms_last_epoch']:.4f} | {time.time() - t0:.0f}s")
    ev = score(va, va.rows, oof)
    np.save(out / f"oof_idx_{a.val}.npy", oof)
    res = {"args": vars(a), "oof": ev, "folds": folds, "val_best": ev["pdms"], "val_epoch0": ev["pdms_ck2"]}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    log(f"OOF {a.val}: PDMS {ev['pdms']:.4f} (CK2 {ev['pdms_ck2']:.4f}, d {100 * ev['delta']:+.2f} CI "
        f"{100 * ev['delta_ci95'][0]:+.2f} {100 * ev['delta_ci95'][1]:+.2f}) changed {ev['frac_changed']:.3f} "
        f"NC {ev['nc']:.4f}/{ev['nc_ck2']:.4f} DAC {ev['dac']:.4f}/{ev['dac_ck2']:.4f} EP {ev['ep']:.4f}/{ev['ep_ck2']:.4f} "
        f"TTC {ev['ttc']:.4f}/{ev['ttc_ck2']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
