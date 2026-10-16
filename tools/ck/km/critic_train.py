#!/usr/bin/env python
"""km: critic STRUCTURE experiments on the frozen CK2 v2 planner.  Trained on navtrain_train ONLY (val = pre-declared
choice + monitoring, navtest = once).

  cd ~/kyungmin/SSR-ck2km && source km_env.sh && CUDA_VISIBLE_DEVICES=0 $PY tools/ck/km/critic_train.py \
      --run-dir <RUN> --out <dir> --arm base|td [--fold all|A|B] [--epochs E]
  ... --eval navtrain_val,navtest            (dump logits of a trained critic; rules / scores: critic_eval.py)

Common to every arm (so arms differ only in the structure under test):
  scene  = frozen CK2 v2 agent (eval, no grad) -> bev_embed
  critic = FRESH CK2-architecture critic (online2.build_student_ck2: CKNet 'S', lon / gate frozen), score-head bias =
           logit(train label mean); scores the 96 SUBMITTED trajectories (CK2's lateral correction, dumped lat_traj)
  label  = BCE of the 5 CK_KEYS logits vs the official per-trajectory labels of those trajectories (lat96 labels)
  deploy = CK2's rule with the new probabilities: select2.select96(v2_final, v2_im, sigmoid(logits), beta 0.5, set all)

arm base, --fold A|B (proposal 1, fold-split critics): the same critic trained on one of two log halves of
  navtrain_train (logs shuffled with --seed, alternate).  Two halves = two critics with independent training data;
  critic_eval.py combines them (avg / min / select-by-one-verify-by-other) against the winner's curse.
arm td (proposal 2, frame-to-frame bootstrapped critic): for a token t whose previous log frame p (0.5 s earlier,
  scene.frames[2]) is also a training token, the critic at p scores p's 96 submitted trajectories; each one's
  continuation (poses 1.0 .. 4.0 s of p, plus one constant-velocity pose, moved into t's frame with the recorded
  relative ego pose) is scored by the critic at t (a later observation) as extra candidates, and
      L_td = sum_j w_j BCE(logit_p[j], stopgrad(sigmoid(logit_t(cont_j)))) / sum_j w_j,
      w_j  = exp(-|cand_j(0.5 s) - ego_t|^2 / sigma^2)    (only plans the ego actually followed are bootstrapped).
  The label loss is applied to t only (same label exposure per epoch as base); p contributes only through L_td.
"""
from __future__ import annotations

import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[3])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from navsim.agents.para_ssr.ck import select2 as S2  # noqa: E402
from navsim.agents.para_ssr.ck.constants import CK_LABEL_IDX, LBL  # noqa: E402
from navsim.agents.para_ssr.ck.online import _sanitize_traj, bev_sgrid  # noqa: E402
from tools.ck.km.ft_trunk import Pool, build, log  # noqa: E402

P_PDMS = LBL["pdms"]
BETA = 0.5


# ----------------------------------------------------------------------------------------------- data
def fold_tokens(P: Pool, fold: str, seed: int):
    toks = P.tokens.token.astype(str).to_numpy()[P.rows]
    logs = P.tokens.log.astype(str).to_numpy()[P.rows]
    if fold == "all":
        return toks
    u = np.unique(logs)
    u = u[np.random.default_rng(seed).permutation(len(u))]
    half = set(u[0::2]) if fold == "A" else set(u[1::2])
    return toks[np.isin(logs, list(half))]


class PairDS(torch.utils.data.Dataset):
    """token -> (token, features); with prev=True also the previous log frame's token / features / relative pose
    (p's ego pose in t's frame, AgentInput.ego_statuses[2]) when that frame is in `pool`."""

    def __init__(self, loader, builders, tokens, prev: bool, pool: set):
        self.loader, self.builders, self.tokens, self.prev, self.pool = loader, list(builders), list(tokens), prev, pool

    def __len__(self):
        return len(self.tokens)

    def _feat(self, ai):
        f = {}
        for b in self.builders:
            f.update(b.compute_features(ai))
        return f

    def __getitem__(self, i):
        tok = self.tokens[i]
        scene = self.loader.get_scene_from_token(tok)
        ai = scene.get_agent_input()
        out = {"tok": tok, "f": self._feat(ai), "ptok": None}
        if self.prev:
            ptok = scene.frames[2].token
            if ptok in self.pool:
                pai = self.loader.get_scene_from_token(ptok).get_agent_input()
                out.update(ptok=ptok, pf=self._feat(pai),
                           pose=torch.as_tensor(np.asarray(ai.ego_statuses[2].ego_pose, np.float32)))
        return out


def collate(batch):
    st = lambda fs: {k: torch.stack([f[k] for f in fs]) for k in fs[0]}
    out = {"tok": [b["tok"] for b in batch], "f": st([b["f"] for b in batch])}
    has = [i for i, b in enumerate(batch) if b["ptok"] is not None]
    out["pi"] = has
    if has:
        out["ptok"] = [batch[i]["ptok"] for i in has]
        out["pf"] = st([batch[i]["pf"] for i in has])
        out["pose"] = torch.stack([batch[i]["pose"] for i in has])
    return out


def make_loader(split: str, tokens, agent, bs: int, workers: int, prev: bool, shuffle_seed=None):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader
    from tools.ck.data import common as CM
    sp = CM.split(split)
    tdf = CM.split_tokens(split)
    tdf = tdf[tdf.token.isin(set(tokens))]
    sf = instantiate(OmegaConf.load(CM.resolve_path(sp["scene_filter"])))
    sf = replace(sf, tokens=list(tokens), max_scenes=None)
    need = set(tdf.log)
    sf = replace(sf, log_names=sorted(set(sf.log_names or need) & need))
    ld = SceneLoader(data_path=Path(sp["navsim_logs"]), sensor_blobs_path=Path(sp["sensor_blobs"]), scene_filter=sf,
                     sensor_config=agent.get_sensor_config())
    toks = list(tokens)
    if shuffle_seed is not None:
        toks = list(np.random.default_rng(shuffle_seed).permutation(toks))
    ds = PairDS(ld, agent.get_feature_builders(), toks, prev, set(tokens))
    return torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=workers, collate_fn=collate, pin_memory=True,
                                       prefetch_factor=2 if workers else None, persistent_workers=False)


def to_dev(f, dev):
    return {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else v.dtype, non_blocking=True)
            for k, v in f.items()}


# ----------------------------------------------------------------------------------------------- critic
def build_critic(agent, dev, prior):
    from navsim.agents.para_ssr.ck.online2 import build_student_ck2
    net = build_student_ck2(agent._ck_e2e2.cfg, apply_prior=False).to(dev)
    net.set_score_prior(prior)
    return net


def critic_logits(ck, bev, traj, status, extra=None):
    """bev [B, 5000, 256], traj [B, K, 8, 3] -> logits [B, K, 5] (and [B, K2, 5] for extra)."""
    from navsim.agents.para_ssr.refiner.e2e import ego_inputs
    B, K = traj.shape[:2]
    ego = ego_inputs(status)
    fb = torch.nan_to_num(traj[:, 0], nan=0.0)
    tr, _ = _sanitize_traj(traj, fb)
    with torch.autocast(device_type=bev.device.type, enabled=False):
        feat, mem = ck.trunk.scene(bev_sgrid(bev.float()), B)
        if extra is None:
            return ck.score(ck.trunk.candidates(feat, mem, tr, *ego), B, K), None
        ex, _ = _sanitize_traj(extra, fb)
        o = ck.trunk.candidates(feat, mem, torch.cat([tr, ex], 1), *ego)
        s = ck.score(o, B, K + extra.shape[1])
        return s[:, :K], s[:, K:]


def continuation(traj_p, pose):
    """traj_p [b, K, 8, 3] (p frame, t = 0.5 .. 4.0 s), pose [b, 3] = p's ego pose in t's frame ->
    (cont [b, K, 8, 3] in t's frame for t = 0.5 .. 4.0 s after t, d [b, K] = |cand(0.5 s of p) - ego at t|)."""
    c, s = torch.cos(pose[:, 2])[:, None, None], torch.sin(pose[:, 2])[:, None, None]
    x, y, h = traj_p[..., 0], traj_p[..., 1], traj_p[..., 2]
    X = c * x - s * y + pose[:, 0, None, None]
    Y = s * x + c * y + pose[:, 1, None, None]
    H = h + pose[:, 2, None, None]
    q = torch.stack([X, Y, H], -1)                                    # p's plan in t's frame
    d = q[:, :, 0, :2].norm(dim=-1)                                   # p's 0.5 s pose vs t's origin
    last = q[:, :, 7] + (q[:, :, 7] - q[:, :, 6])
    cont = torch.cat([q[:, :, 1:8], last[:, :, None]], 2)
    return cont, d


# ----------------------------------------------------------------------------------------------- eval dump
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
        lg[rows] = critic_logits(ck, bev, d["lat"], d["status"])[0].float().cpu().numpy()
    np.save(out / f"logits_{split}.npy", lg)
    np.save(out / f"rows_{split}.npy", P.rows)
    prob = 1.0 / (1.0 + np.exp(-lg[P.rows].astype(np.float64)))
    idx = S2.select96(P.v2_final[P.rows], P.v2_im[P.rows], prob, P.valid[P.rows], BETA)
    ar = np.arange(len(P.rows))
    base = S2.select96(P.v2_final[P.rows], P.v2_im[P.rows], 1 / (1 + np.exp(-np.asarray(P.logit[P.rows], np.float64))),
                       P.valid[P.rows], BETA)
    lab = P.lab_lat[P.rows]
    return {"n": int(len(P.rows)), "pdms": float(lab[ar, idx, P_PDMS].mean()),
            "pdms_ck2": float(lab[ar, base, P_PDMS].mean()), "frac_changed": float(np.mean(idx != base))}


# ----------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run", default="ck2e2e_bevkd")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="base", choices=("base", "td"))
    ap.add_argument("--fold", default="all", choices=("all", "A", "B"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--td-weight", type=float, default=1.0)
    ap.add_argument("--td-sigma", type=float, default=0.3, help="m")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-eval", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", default="", help="comma list of splits: dump logits of ckpt_last.pt and exit")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0")
    torch.manual_seed(a.seed)
    run_dir = Path(a.run_dir)
    root = run_dir / "eval" / "root"
    agent, _ck2, ckpt = build(run_dir, dev)
    del _ck2
    tr = Pool(root, a.run, "navtrain_train", need_pool=False)
    lab_k = np.asarray(tr.lab_lat[tr.rows][..., list(CK_LABEL_IDX)], np.float64)
    allowed_tr = tr.allowed[tr.rows]
    prior = np.array([np.nanmean(lab_k[..., k][allowed_tr]) for k in range(5)])
    ck = build_critic(agent, dev, prior)

    if a.eval:
        st = torch.load(out / "ckpt_last.pt", map_location=dev)
        ck.load_state_dict(st["ck"])
        res = json.loads((out / "result.json").read_text()) if (out / "result.json").is_file() else {}
        for sp in a.eval.split(","):
            P = Pool(root, a.run, sp, need_pool=False)
            if a.limit_eval:
                P.rows = P.rows[:a.limit_eval]
            ev = dump_logits(agent, ck, P, sp, dev, 8, a.workers, out)
            res.setdefault("eval", {})[sp] = ev | {"epoch": st["epoch"]}
            log(f"EVAL {sp} (epoch {st['epoch']}): PDMS {ev['pdms']:.4f} (CK2 {ev['pdms_ck2']:.4f}) changed "
                f"{ev['frac_changed']:.3f} n {ev['n']}")
        (out / "result.json").write_text(json.dumps(res, indent=1))
        return 0

    toks = fold_tokens(tr, a.fold, a.seed)
    if a.limit_train:
        toks = toks[:a.limit_train]
    log(f"arm {a.arm} fold {a.fold}: {len(toks)} train tokens, prior {np.round(prior, 4).tolist()}")
    params = [p for p in ck.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    n_steps = max(a.epochs * (len(toks) // (a.bs * a.accum)), 8)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=n_steps, pct_start=max(0.05, 3.0 / n_steps))
    ep0, step = 0, 0
    if (out / "ckpt_last.pt").is_file():                      # resume (end of an epoch)
        st = torch.load(out / "ckpt_last.pt", map_location=dev)
        ck.load_state_dict(st["ck"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        ep0, step = st["epoch"], st["step"]
        log(f"resumed after epoch {ep0} (step {step})")
    hist = json.loads((out / "hist.json").read_text()) if (out / "hist.json").is_file() and ep0 else []
    lab_idx = torch.as_tensor(list(CK_LABEL_IDX), device=dev)
    for ep in range(ep0 + 1, a.epochs + 1):
        ck.train()
        dl = make_loader("navtrain_train", toks, agent, a.bs, a.workers, prev=a.arm == "td",
                         shuffle_seed=a.seed * 1000 + ep)
        t0, acc, mb = time.time(), {"bce": [], "td": [], "tdw": []}, 0
        opt.zero_grad(set_to_none=True)
        for b in dl:
            rows = np.array([tr.row_of[t] for t in b["tok"]])
            d = tr.get(rows, dev)
            with torch.no_grad():
                bev = agent(to_dev(b["f"], dev))["bev_embed"].detach()
            y = torch.as_tensor(np.asarray(tr.lab_lat[rows]), device=dev).float().index_select(-1, lab_idx)
            m = (d["allowed"][..., None] & torch.isfinite(y)).float()
            td_on = a.arm == "td" and len(b["pi"]) > 0
            if td_on:
                prow = np.array([tr.row_of[t] for t in b["ptok"]])
                dp = tr.get(prow, dev)
                with torch.no_grad():
                    bev_p = agent(to_dev(b["pf"], dev))["bev_embed"].detach()
                cont, dist = continuation(dp["lat"], b["pose"].to(dev))
                pi = torch.as_tensor(b["pi"], device=dev)
                ex = d["lat"].clone()
                ex[pi] = cont                                   # rows without a previous frame: unused dummies
                lg_t, lg_c = critic_logits(ck, bev, d["lat"], d["status"], extra=ex)
                lg_c = lg_c.index_select(0, pi)
                lg_p, _ = critic_logits(ck, bev_p, dp["lat"], dp["status"])
                w = torch.exp(-(dist / a.td_sigma) ** 2) * dp["allowed"].float()
                w = w * torch.isfinite(cont).flatten(2).all(-1).float()
                tgt = torch.sigmoid(lg_c).detach()
                bce_td = F.binary_cross_entropy_with_logits(lg_p, tgt, reduction="none").mean(-1)     # [b, K]
                L_td = (bce_td * w).sum() / w.sum().clamp_min(1e-6)
                acc["tdw"].append(float(w.sum() / len(b["pi"])))
            else:
                lg_t, _ = critic_logits(ck, bev, d["lat"], d["status"])
                L_td = lg_t.new_zeros(())
            bce = (F.binary_cross_entropy_with_logits(lg_t, torch.nan_to_num(y), reduction="none") * m).sum() / \
                m.sum().clamp_min(1.0)
            loss = (bce + a.td_weight * L_td) / a.accum
            if not torch.isfinite(loss):
                log("non-finite loss: micro-batch skipped")
                continue
            loss.backward()
            acc["bce"].append(float(bce)); acc["td"].append(float(L_td))
            mb += 1
            if mb % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                if step < n_steps - 1:
                    sched.step()
                step += 1
                if step % 100 == 0:
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
    (out / "result.json").write_text(json.dumps({"args": vars(a), "v2_ckpt": str(ckpt), "n_train": int(len(toks)),
                                                 "prior": prior.tolist(), "hist": hist}, indent=1))
    log("train done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
