#!/usr/bin/env python
"""km: structures ADDED ON TOP OF CK2 (control = CK2 itself, navtest 92.17 / 90.74).  navtrain_train only; v2 frozen
(cached bev_embed, dump_bev.py); val = pre-declared choice; navtest once.

  cd ~/kyungmin/SSR-ck2km && source km_env.sh && CUDA_VISIBLE_DEVICES=0 $PY tools/ck/km/critic_ck2.py \
      --run-dir <RUN> --bev-cache <dir> --out <dir> --arm ft|td|indep [--epochs E --lr LR]
  ... --eval navtrain_val,navtest        (dump logits; live v2 forward; rules / scores: critic_eval.py)

Every arm scores CK2's 96 pool columns exactly as CK2 deploys them (logits on cand96 -> select2.select96 with the
v2 blend, beta 0.5, set all; the submitted trajectory = CK2's lateral correction of the selected column), and its label
is the official score of that submitted trajectory (lat96 labels).
  ft    : CK2's critic, further trained with the label BCE only        (control for td: same steps, same lr)
  td    : CK2's critic + proposal 2, the frame-to-frame bootstrapped consistency term (critic_train.py docstring):
          L_td = sum_j w_j BCE(logit_p[j], sg(sigmoid(logit_t(continuation_j)))) / sum_j w_j, w_j = exp(-(d_j/0.3 m)^2)
  indep : proposal 1's second estimator: a FRESH critic of CK2's architecture trained like CK2's (30 epochs) on the
          same data; critic_eval.py --pair CK2 + indep: avg (ensemble control), min, xA / xB (select by one critic's
          top-8, verify by the other).
"""
from __future__ import annotations

import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[3])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from navsim.agents.para_ssr.ck import select2 as S2  # noqa: E402
from navsim.agents.para_ssr.ck.constants import CK_LABEL_IDX, LBL  # noqa: E402
from tools.ck.km.critic_train import build_critic, continuation, critic_logits, make_loader, to_dev  # noqa: E402
from tools.ck.km.ft_trunk import Pool, build, log  # noqa: E402

P_PDMS = LBL["pdms"]
BETA = 0.5


def prev_table(P: Pool, split: str, d: Path):
    """prev_row [N] (row of scene.frames[2] if it is a token of the split, else -1), pose [N, 3] (that frame's ego pose
    in the token's frame, AgentInput.ego_statuses[2]).  Cached in d/prev.npz."""
    f = d / "prev.npz"
    if f.is_file():
        z = np.load(f)
        return z["prev_row"], z["pose"]
    from dataclasses import replace
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader
    from tools.ck.data import common as CM
    sp = CM.split(split)
    toks = P.tokens.token.astype(str).to_numpy()[P.rows]
    sf = instantiate(OmegaConf.load(CM.resolve_path(sp["scene_filter"])))
    sf = replace(sf, tokens=list(toks), max_scenes=None,
                 log_names=sorted(set(P.tokens.log.astype(str).to_numpy()[P.rows])))
    ld = SceneLoader(data_path=Path(sp["navsim_logs"]), sensor_blobs_path=Path(sp["sensor_blobs"]), scene_filter=sf,
                     sensor_config=SensorConfig.build_no_sensors())
    N = len(P.tokens)
    prev, pose = np.full(N, -1, np.int64), np.zeros((N, 3), np.float32)
    for i, r in enumerate(P.rows):
        sc = ld.get_scene_from_token(toks[i])
        q = P.row_of.get(sc.frames[2].token, -1)
        if q >= 0:
            prev[r] = q
            pose[r] = np.asarray(sc.get_agent_input().ego_statuses[2].ego_pose, np.float32)
        if i % 10000 == 0:
            log(f"prev table {i}/{len(P.rows)}")
    np.savez(f, prev_row=prev, pose=pose)
    log(f"prev table: {(prev[P.rows] >= 0).sum()} / {len(P.rows)} tokens have their previous frame")
    return prev, pose


class CachedDS(torch.utils.data.Dataset):
    def __init__(self, bev_path, rows, prev, pose, td: bool):
        self.bev_path, self.rows, self.prev, self.pose, self.td = bev_path, rows, prev, pose, td
        self.bev = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.bev is None:
            self.bev = np.load(self.bev_path, mmap_mode="r")
        r = self.rows[i]
        out = {"row": r, "bev": torch.from_numpy(np.ascontiguousarray(self.bev[r])), "p": -1}
        if self.td and self.prev[r] >= 0:
            out.update(p=int(self.prev[r]), pbev=torch.from_numpy(np.ascontiguousarray(self.bev[self.prev[r]])),
                       pose=torch.from_numpy(self.pose[r]))
        return out


def collate(batch):
    out = {"rows": np.array([b["row"] for b in batch]), "bev": torch.stack([b["bev"] for b in batch])}
    has = [i for i, b in enumerate(batch) if b["p"] >= 0]
    out["pi"] = has
    if has:
        out["prow"] = np.array([batch[i]["p"] for i in has])
        out["pbev"] = torch.stack([batch[i]["pbev"] for i in has])
        out["pose"] = torch.stack([batch[i]["pose"] for i in has])
    return out


@torch.no_grad()
def dump_logits(agent, ck, P: Pool, split, dev, bs, workers, out: Path):
    ck.eval()
    toks = P.tokens.token.astype(str).to_numpy()[P.rows]
    dl = make_loader(split, toks, agent, bs, workers, prev=False)
    lg = np.full((len(P.tokens), 96, 5), np.nan, np.float32)
    for b in dl:
        bev = agent(to_dev(b["f"], dev))["bev_embed"]
        rows = np.array([P.row_of[t] for t in b["tok"]])
        d = P.get(rows, dev)
        lg[rows] = critic_logits(ck, bev, d["cand"], d["status"])[0].float().cpu().numpy()
    np.save(out / f"logits_{split}.npy", lg)
    np.save(out / f"rows_{split}.npy", P.rows)
    sig = lambda x: 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))
    idx = S2.select96(P.v2_final[P.rows], P.v2_im[P.rows], sig(lg[P.rows]), P.valid[P.rows], BETA)
    base = S2.select96(P.v2_final[P.rows], P.v2_im[P.rows], sig(P.logit[P.rows]), P.valid[P.rows], BETA)
    ar = np.arange(len(P.rows))
    lab = P.lab_lat[P.rows]
    return {"n": int(len(P.rows)), "pdms": float(lab[ar, idx, P_PDMS].mean()),
            "pdms_ck2": float(lab[ar, base, P_PDMS].mean()), "frac_changed": float(np.mean(idx != base))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run", default="ck2e2e_bevkd")
    ap.add_argument("--bev-cache", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, choices=("ft", "td", "indep"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--td-weight", type=float, default=1.0)
    ap.add_argument("--td-sigma", type=float, default=0.3, help="m")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-eval", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", default="", help="comma list of splits ('ckpt0' = CK2 itself for ft / td)")
    ap.add_argument("--eval-ckpt0", action="store_true", help="dump the untrained start (CK2 for ft / td)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0")
    torch.manual_seed(a.seed)
    run_dir = Path(a.run_dir)
    root = run_dir / "eval" / "root"
    agent, ck2, ckpt = build(run_dir, dev)
    tr = Pool(root, a.run, "navtrain_train", need_pool=False)
    if a.arm == "indep":
        lab_k = np.asarray(tr.lab_lat[tr.rows][..., list(CK_LABEL_IDX)], np.float64)
        al = tr.allowed[tr.rows]
        prior = np.array([np.nanmean(lab_k[..., k][al]) for k in range(5)])
        ck = build_critic(agent, dev, prior)
        del ck2
    else:
        ck = ck2

    if a.eval:
        if not a.eval_ckpt0:
            st = torch.load(out / "ckpt_last.pt", map_location=dev)
            ck.load_state_dict(st["ck"])
        ep = 0 if a.eval_ckpt0 else st["epoch"]
        o = out / "ckpt0" if a.eval_ckpt0 else out
        o.mkdir(parents=True, exist_ok=True)
        res = json.loads((o / "result.json").read_text()) if (o / "result.json").is_file() else {}
        for sp in a.eval.split(","):
            P = Pool(root, a.run, sp, need_pool=False)
            if a.limit_eval:
                P.rows = P.rows[:a.limit_eval]
            ev = dump_logits(agent, ck, P, sp, dev, 8, 6, o)
            res.setdefault("eval", {})[sp] = ev | {"epoch": ep}
            log(f"EVAL {sp} (epoch {ep}): PDMS {ev['pdms']:.4f} (CK2 {ev['pdms_ck2']:.4f}) changed "
                f"{ev['frac_changed']:.3f} n {ev['n']}")
        (o / "result.json").write_text(json.dumps(res, indent=1))
        return 0

    del agent
    torch.cuda.empty_cache()
    bev_path = Path(a.bev_cache) / "navtrain_train" / "bev.npy"
    dn = np.zeros(len(tr.tokens), bool)
    for f in (Path(a.bev_cache) / "navtrain_train").glob("done_*.npy"):
        dn |= np.load(f)
    rows = tr.rows[dn[tr.rows]]
    assert len(rows) == len(tr.rows), f"bev cache incomplete: {len(rows)} / {len(tr.rows)}"
    prev, pose = prev_table(tr, "navtrain_train", Path(a.bev_cache) / "navtrain_train")
    if a.limit_train:
        rows = rows[:a.limit_train]
    td = a.arm == "td"
    log(f"arm {a.arm}: {len(rows)} train tokens, {(prev[rows] >= 0).sum()} with previous frame, lr {a.lr}, "
        f"epochs {a.epochs}")
    params = [p for p in ck.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    n_steps = max(a.epochs * (len(rows) // (a.bs * a.accum)), 8)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=n_steps, pct_start=max(0.05, 3.0 / n_steps))
    ep0, step = 0, 0
    if (out / "ckpt_last.pt").is_file():
        st = torch.load(out / "ckpt_last.pt", map_location=dev)
        ck.load_state_dict(st["ck"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        ep0, step = st["epoch"], st["step"]
        log(f"resumed after epoch {ep0} (step {step})")
    hist = json.loads((out / "hist.json").read_text()) if (out / "hist.json").is_file() and ep0 else []
    lab_idx = list(CK_LABEL_IDX)
    for ep in range(ep0 + 1, a.epochs + 1):
        ck.train()
        perm = np.random.default_rng(a.seed * 1000 + ep).permutation(rows)
        dl = torch.utils.data.DataLoader(CachedDS(bev_path, perm, prev, pose, td), batch_size=a.bs,
                                         num_workers=a.workers, collate_fn=collate, pin_memory=True,
                                         prefetch_factor=4 if a.workers else None)
        t0, acc, mb = time.time(), {"bce": [], "td": [], "tdw": []}, 0
        opt.zero_grad(set_to_none=True)
        for b in dl:
            r = b["rows"]
            d = tr.get(r, dev)
            bev = b["bev"].to(dev, non_blocking=True).float()
            y = torch.as_tensor(np.asarray(tr.lab_lat[r])[..., lab_idx], device=dev).float()
            m = (d["allowed"][..., None] & torch.isfinite(y)).float()
            if td and b["pi"]:
                dp = tr.get(b["prow"], dev)
                cont, dist = continuation(dp["cand"], b["pose"].to(dev))
                pi = torch.as_tensor(b["pi"], device=dev)
                ex = d["cand"].clone()
                ex[pi] = cont
                lg_t, lg_c = critic_logits(ck, bev, d["cand"], d["status"], extra=ex)
                lg_c = lg_c.index_select(0, pi)
                lg_p, _ = critic_logits(ck, b["pbev"].to(dev, non_blocking=True).float(), dp["cand"], dp["status"])
                w = torch.exp(-(dist / a.td_sigma) ** 2) * dp["allowed"].float()
                w = w * torch.isfinite(cont).flatten(2).all(-1).float()
                bce_td = F.binary_cross_entropy_with_logits(lg_p, torch.sigmoid(lg_c).detach(), reduction="none").mean(-1)
                L_td = (bce_td * w).sum() / w.sum().clamp_min(1e-6)
                acc["tdw"].append(float(w.sum() / len(b["pi"])))
            else:
                lg_t, _ = critic_logits(ck, bev, d["cand"], d["status"])
                L_td = lg_t.new_zeros(())
            bce = (F.binary_cross_entropy_with_logits(lg_t, torch.nan_to_num(y), reduction="none") * m).sum() / \
                m.sum().clamp_min(1.0)
            loss = (bce + a.td_weight * L_td) / a.accum
            if not torch.isfinite(loss):
                log("non-finite loss: micro-batch skipped")
                continue
            loss.backward()
            acc["bce"].append(float(bce.detach())); acc["td"].append(float(L_td.detach()))
            mb += 1
            if mb % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                if step < n_steps - 1:
                    sched.step()
                step += 1
                if step % 200 == 0:
                    el = time.time() - t0
                    log(f"ep {ep} step {step}/{n_steps} | bce {np.mean(acc['bce'][-800:]):.4f} td "
                        f"{np.mean(acc['td'][-800:]):.4f} w/pair {np.mean(acc['tdw'][-800:]) if acc['tdw'] else 0:.2f}"
                        f" | {mb * a.bs / el:.1f} tok/s")
        hist.append({"epoch": ep, "bce": float(np.mean(acc["bce"])), "td": float(np.mean(acc["td"])),
                     "sec": round(time.time() - t0)})
        tmp = out / "ckpt_last.pt.tmp"
        torch.save({"ck": ck.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                    "step": step, "args": vars(a)}, tmp)
        os.replace(tmp, out / "ckpt_last.pt")
        (out / "hist.json").write_text(json.dumps(hist, indent=1))
        log(f"epoch {ep}: bce {hist[-1]['bce']:.4f} td {hist[-1]['td']:.4f} | {hist[-1]['sec']}s")
    (out / "result.json").write_text(json.dumps({"args": vars(a), "v2_ckpt": str(ckpt), "n_train": int(len(rows)),
                                                 "hist": hist}, indent=1))
    log("train done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
