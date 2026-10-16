#!/usr/bin/env python
"""km stage 1.5-CV: CK2 critic (trunk + score head) + listwise head fine-tuned on HELD-OUT navtrain_val folds only.

Same model / loss as ft_trunk.py (v2 frozen, dumped CK2 pool + CK2's lateral correction, listwise official PDMS
+ kd * BCE to the frozen CK2 logits), but trained K-fold on navtrain_val (folds by log, identical to listwise_cv.py)
because v2 / CK are in-sample on navtrain_train.  Fold f: the listwise head starts from <head-init>/fold{f}.pt (the
listwise_cv.py head of the SAME training folds -> no held-out data seen), CK from the CK2 weights; fixed epochs; then
it selects on fold f (out-of-fold).

  train folds (can be split over GPUs):  ft_cv.py --run-dir <RUN> --out <dir> --head-init <cv dir> --folds 0,1,2
  merge (OOF over all val):              ft_cv.py --run-dir <RUN> --out <dir> --head-init <cv dir> --merge
  navtest ONCE (mean delta of K models): ft_cv.py ... --test navtest --test-only
"""
from __future__ import annotations

import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[3])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from navsim.agents.para_ssr.ck import listwise as LW  # noqa: E402
from tools.ck.km import ft_trunk as FT  # noqa: E402
from tools.ck.km.ft_trunk import Pool, log  # noqa: E402
from tools.ck.km.listwise_cv import score  # noqa: E402


def folds_of(P: Pool, k: int, seed: int) -> np.ndarray:
    """== listwise_cv.fold_of_rows (same logs -> same folds)."""
    logs = np.asarray(P.tokens["log"].astype(str))[P.rows]
    u = np.unique(logs)
    np.random.default_rng(seed).shuffle(u)
    f = {lg: i % k for i, lg in enumerate(u)}
    return np.array([f[lg] for lg in logs])


def make_head(hargs, dev):
    return LW.ListwiseHead(d=hargs["d"], n_layers=hargs["layers"], dropout=hargs["dropout"],
                           cross=bool(hargs["cross"])).to(dev)


@torch.no_grad()
def deltas(agent, ck, head, P: Pool, split, rows, dev, workers) -> dict:
    ck.eval(); head.eval()
    toks = P.tokens.token.astype(str).to_numpy()[rows]
    out = {}
    for btok, feats, _gt in FT.loader_for(split, toks, agent, 8, workers):
        fg = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else v.dtype, non_blocking=True)
              for k, v in feats.items()}
        bev = agent(fg)["bev_embed"]
        rr = np.array([P.row_of[t] for t in btok])
        d, _, _ = FT.ck_forward(ck, head, bev, P.get(rr, dev))
        for r, x in zip(rr, d.double().cpu().numpy()):
            out[int(r)] = x
    return out


def train_fold(a, agent, ck0_state, ck, P: Pool, rows, hargs, head_init, dev):
    ck.load_state_dict(ck0_state)
    head = make_head(hargs, dev)
    head.load_state_dict(torch.load(head_init, map_location=dev))
    toks = P.tokens.token.astype(str).to_numpy()[rows]
    params = [{"params": [p for p in ck.parameters() if p.requires_grad], "lr": a.lr},
              {"params": head.parameters(), "lr": a.lr_head}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    n_steps = a.epochs * (len(toks) // (a.bs * a.accum))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr, a.lr_head], total_steps=max(n_steps, 1), pct_start=0.05)
    step, hist = 0, []
    for ep in range(1, a.epochs + 1):
        ck.train(); head.train()
        t0, acc, mb = time.time(), {"exp": [], "kd": []}, 0
        opt.zero_grad(set_to_none=True)
        for btok, feats, _gt in FT.loader_for(a.val, toks, agent, a.bs, a.workers, shuffle_seed=a.seed + ep):
            fg = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else v.dtype, non_blocking=True)
                  for k, v in feats.items()}
            with torch.no_grad():
                bev = agent(fg)["bev_embed"].detach()
            b = P.get(np.array([P.row_of[t] for t in btok]), dev)
            delta, lg, lg_lat = FT.ck_forward(ck, head, bev, b)
            L = LW.listwise_loss(b["s0"].float() + delta, b["pdms"], b["allowed"], a.tau)
            m = b["allowed"][..., None].float()
            kd = sum((F.binary_cross_entropy_with_logits(x, torch.sigmoid(y), reduction="none") * m).sum()
                     for x, y in ((lg, b["logit0"]), (lg_lat, b["logit0_lat"]))) / (m.sum() * 5)
            loss = (L["loss"] + a.kd * kd) / a.accum
            if not torch.isfinite(loss):
                log("non-finite loss: micro-batch skipped")
                continue
            loss.backward()
            acc["exp"].append(float(L["exp_pdms"])); acc["kd"].append(float(kd))
            mb += 1
            if mb % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(list(ck.parameters()) + list(head.parameters()), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                if step < n_steps - 1:
                    sched.step()
                step += 1
                if step % 100 == 0:
                    log(f"  ep {ep} step {step}/{n_steps} | E[PDMS] {np.mean(acc['exp'][-800:]):.4f} kd "
                        f"{np.mean(acc['kd'][-800:]):.4f} | {mb * a.bs / (time.time() - t0):.1f} tok/s")
        hist.append({"epoch": ep, "train_exp_pdms": float(np.mean(acc["exp"])), "kd": float(np.mean(acc["kd"])),
                     "sec": round(time.time() - t0)})
        log(f"  ep {ep}: train E[PDMS] {hist[-1]['train_exp_pdms']:.4f} kd {hist[-1]['kd']:.4f} | {hist[-1]['sec']}s")
    return head, hist


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run", default="ck2e2e_bevkd")
    ap.add_argument("--out", required=True)
    ap.add_argument("--head-init", required=True, help="listwise_cv.py output dir (fold{f}.pt + result.json)")
    ap.add_argument("--val", default="navtrain_val")
    ap.add_argument("--test", default="")
    ap.add_argument("--folds", default="", help="comma list of folds to train (default all)")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--test-only", action="store_true")
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit-val", type=int, default=0, help="smoke only")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hres = json.loads((Path(a.head_init) / "result.json").read_text())
    hargs = hres["args"]
    a.tau = hargs["tau"]
    k = hargs["k"]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = Path(a.run_dir) / "eval" / "root"
    if a.merge:
        va = Pool(root, a.run, a.val, need_pool=True)
        fold = folds_of(va, k, hargs["seed"])
        oof = np.zeros(len(va.rows), np.int64)
        per = []
        for f in range(k):
            z = np.load(out / f"fold{f}_oof.npz")
            ho = np.flatnonzero(fold == f)
            assert np.array_equal(z["rows"], va.rows[ho]), f"fold {f} rows differ"
            oof[ho] = z["idx"]
            per.append(score(va, va.rows[ho], z["idx"]) | {"fold": f})
        ev = score(va, va.rows, oof)
        res = {"args": vars(a), "head_init_oof": hres["oof"], "oof": ev, "folds": per, "val_best": ev["pdms"],
               "val_epoch0": ev["pdms_ck2"]}
        (out / "result.json").write_text(json.dumps(res, indent=1))
        log(f"OOF {a.val}: PDMS {ev['pdms']:.4f} (CK2 {ev['pdms_ck2']:.4f}, d {100 * ev['delta']:+.2f} CI "
            f"{100 * ev['delta_ci95'][0]:+.2f} {100 * ev['delta_ci95'][1]:+.2f}; head only "
            f"{hres['oof']['pdms']:.4f}) changed {ev['frac_changed']:.3f} NC {ev['nc']:.4f} DAC {ev['dac']:.4f} "
            f"EP {ev['ep']:.4f} TTC {ev['ttc']:.4f}")
        return 0
    agent, ck, ckpt = FT.build(Path(a.run_dir), dev)
    ck0 = copy.deepcopy(ck.state_dict())
    if a.test_only:
        te = Pool(root, a.run, a.test, need_pool=True)
        acc = None
        for f in range(k):
            st = torch.load(out / f"fold{f}.pt", map_location=dev)
            ck.load_state_dict(st["ck"])
            head = make_head(hargs, dev)
            head.load_state_dict(st["head"])
            d = deltas(agent, ck, head, te, a.test, te.rows, dev, a.workers)
            D = np.stack([d[int(r)] for r in te.rows])
            acc = D if acc is None else acc + D
            log(f"test fold model {f} done")
        idx = LW.select(te.s0[te.rows] + acc / k, te.allowed[te.rows])
        ev = score(te, te.rows, idx)
        np.save(out / f"idx_{a.test}.npy", idx); np.save(out / f"rows_{a.test}.npy", te.rows)
        res = json.loads((out / "result.json").read_text())
        res["test"] = ev
        (out / "result.json").write_text(json.dumps(res, indent=1))
        log(f"TEST {a.test} ({k} fold models, mean delta): PDMS {ev['pdms']:.4f} (CK2 {ev['pdms_ck2']:.4f}, d "
            f"{100 * ev['delta']:+.2f} CI {100 * ev['delta_ci95'][0]:+.2f} {100 * ev['delta_ci95'][1]:+.2f}) "
            f"changed {ev['frac_changed']:.3f} n {ev['n']}")
        return 0
    torch.manual_seed(a.seed)
    va = Pool(root, a.run, a.val, need_pool=True)
    fold = folds_of(va, k, hargs["seed"])
    for f in ([int(x) for x in a.folds.split(",")] if a.folds else range(k)):
        tr_rows, ho_rows = va.rows[fold != f], va.rows[fold == f]
        if a.limit_val:
            tr_rows, ho_rows = tr_rows[:a.limit_val], ho_rows[:max(a.limit_val // 4, 8)]
        log(f"fold {f}: train {len(tr_rows)} held-out {len(ho_rows)}")
        head, hist = train_fold(a, agent, ck0, ck, va, tr_rows, hargs, Path(a.head_init) / f"fold{f}.pt", dev)
        torch.save({"ck": ck.state_dict(), "head": head.state_dict(), "args": vars(a), "hist": hist}, out / f"fold{f}.pt")
        d = deltas(agent, ck, head, va, a.val, ho_rows, dev, a.workers)
        idx = LW.select(va.s0[ho_rows] + np.stack([d[int(r)] for r in ho_rows]), va.allowed[ho_rows])
        np.savez(out / f"fold{f}_oof.npz", rows=ho_rows, idx=idx)
        ev = score(va, ho_rows, idx)
        log(f"fold {f}: held-out PDMS {ev['pdms']:.4f} (CK2 {ev['pdms_ck2']:.4f}, d {100 * ev['delta']:+.2f}) "
            f"changed {ev['frac_changed']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
