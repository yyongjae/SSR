#!/usr/bin/env python
"""km stage 1.5: fine-tune the CK2 critic (trunk + score head) together with the listwise head, v2 frozen.

  cd ~/kyungmin/SSR-ck2km && source km_env.sh && CUDA_VISIBLE_DEVICES=1 $PY tools/ck/km/ft_trunk.py \
      --run-dir <RUN> --out <dir> [--kd 1.0] [--tau 0.1] [--epochs 2] [--limit-train N] [--limit-val N]
      [--test navtest --test-only]

Per token (B tokens per micro-batch, accumulate --accum):
  bev      = frozen v2 agent forward (eval, no grad; CK2 inference hook off)  -> bev_embed
  pool     = the DUMPED 96 columns of the frozen CK2 (cand96, lat_traj = CK2's own lateral correction, valid96, v2_final,
             v2_im, CK2 logits) of <RUN>/eval/root -- exactly the trajectories the official labels were computed on
  trainable CK (copy of agent.ck_student): h / logits of cand96 and of lat_traj
  r        = s0 (CK2's deployed score, beta 0.5, frozen) + listwise head(features of all 96 columns)
  loss     = listwise expected official PDMS of the submitted (lateral) trajectory
             + kd * BCE(new logits, sigmoid(CK2 logits)) on pool and lateral columns   (self-distillation anchor)
Selection = argmax r (lateral 'on'); the lateral correction stays CK2's (frozen), so labels == submitted trajectory.
Validation every epoch on --val (live forward, same pipeline); best val epoch kept (ckpt_best.pt).
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
import os  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from navsim.agents.para_ssr.ck import listwise as LW  # noqa: E402
from navsim.agents.para_ssr.ck import select2 as S2  # noqa: E402
from navsim.agents.para_ssr.ck.constants import LBL  # noqa: E402
from navsim.agents.para_ssr.ck.online import _sanitize_traj, bev_sgrid  # noqa: E402
from tools.ck.e2e2 import eval_ck2 as EV  # noqa: E402

P_PDMS = LBL["pdms"]
BETA = 0.5


def log(msg: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


class Pool:
    """dumped pool + official labels of one split (rows = packed tokens.parquet order)."""

    def __init__(self, root: Path, run: str, split: str, need_pool: bool):
        pk, inf = root / "packed" / split, root / "infer" / run / split
        self.tokens = pd.read_parquet(pk / "tokens.parquet")
        n = len(self.tokens)
        ld = lambda d, k: np.load(d / f"{k}.npy", mmap_mode="r")
        self.cand, self.lat, self.valid = ld(pk, "cand96"), ld(inf, "lat_traj"), ld(pk, "valid96")
        self.v2_final, self.v2_im, self.status = ld(pk, "v2_final"), ld(pk, "v2_im"), ld(pk, "status")
        self.logit, self.logit_lat, self.e_lat = ld(inf, "score_logit"), ld(inf, "score_logit_lat"), ld(inf, "e_lat")
        self.lab_lat, ok_lat = EV.load_labels(root / "labels" / split / f"lat96_{run}", n)
        ok_pool = (EV.load_labels(root / "labels" / split / f"pool96_{run}", n)[1] if need_pool
                   else np.ones_like(ok_lat))
        m = EV.token_mask(ld(pk, "ok"), ld(inf, "done"), self.valid, ok_pool, ok_lat)
        self.rows = np.flatnonzero(m)
        self.row_of = {t: i for i, t in enumerate(self.tokens.token.astype(str))}
        self.allowed = S2.allowed_mask(np.asarray(self.valid), S2.VARIANT_SETS["all"])
        self.s0 = S2.pool_scores(np.asarray(self.v2_final), np.asarray(self.v2_im),
                                 1.0 / (1.0 + np.exp(-np.asarray(self.logit, np.float64))), BETA)
        log(f"{split}: {n} tokens, {len(self.rows)} usable")

    def get(self, rows, dev):
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a[rows]), dtype=dt, device=dev)
        return {"cand": t(self.cand), "lat": t(self.lat), "allowed": t(self.allowed, torch.bool),
                "s0": t(self.s0, torch.float64), "v2_final": t(self.v2_final), "v2_im": t(self.v2_im),
                "status": t(self.status), "logit0": t(self.logit), "logit0_lat": t(self.logit_lat),
                "e_lat": t(self.e_lat), "pdms": t(self.lab_lat[:, :, P_PDMS])}


def build(run_dir: Path, dev):
    from navsim.planning.script import run_aux_evaluation as RAE
    from tools.ck.e2e import eval_e2e as E1
    ckpt = E1.resolve_ckpt(run_dir, "last")
    agent = RAE._build_agent(run_dir / "train/code/hydra/config.yaml", ckpt, dev)
    agent.eval()
    for p in agent.parameters():
        p.requires_grad_(False)
    c = agent._ck_e2e2.cfg
    try:
        c.infer_outputs = False
    except Exception:
        object.__setattr__(c, "infer_outputs", False)
    ck = copy.deepcopy(agent.ck_student).to(dev)
    for n, p in ck.named_parameters():       # CK2: lon head zero + frozen, gate head frozen (unused by the score)
        p.requires_grad_(not (n.startswith("trunk.lon_head") or n.startswith("trunk.gate_head")))
    return agent, ck, ckpt


def loader_for(split: str, tokens, agent, bs: int, workers: int, shuffle_seed=None):
    from navsim.common.dataloader import SceneLoader
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from tools.ck.data import common as CM
    from tools.ck.data import dump_v2 as DV
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
    ds = DV.FeatDS(ld, agent.get_feature_builders(), toks)
    return torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=workers, collate_fn=DV.collate, pin_memory=True,
                                       prefetch_factor=2 if workers else None, persistent_workers=False)


def ck_forward(ck, head, bev, b):
    from navsim.agents.para_ssr.refiner.e2e import ego_inputs
    B = b["cand"].shape[0]
    ego = ego_inputs(b["status"])
    cand, _ = _sanitize_traj(b["cand"], b["cand"][:, 0])
    lat, _ = _sanitize_traj(b["lat"], b["cand"][:, 0])
    with torch.autocast(device_type=bev.device.type, enabled=False):
        feat, mem = ck.trunk.scene(bev_sgrid(bev.float()), B)
        both = torch.cat([cand, lat], 1)
        o = ck.trunk.candidates(feat, mem, both, *ego)
        sc = ck.score(o, B, 192)
        h = o["h"].reshape(B, 192, -1)
    x = LW.pool_features(h[:, :96], h[:, 96:], sc[:, :96], sc[:, 96:], b["s0"].float(), b["v2_final"], b["v2_im"],
                         b["e_lat"])
    delta = head(x, b["allowed"])
    return delta, sc[:, :96], sc[:, 96:]


def run_epoch_eval(agent, ck, head, P: Pool, split, dev, bs, workers):
    ck.eval(); head.eval()
    toks = P.tokens.token.astype(str).to_numpy()[P.rows]
    dl = loader_for(split, toks, agent, bs, workers)
    sel, rows_done = [], []
    with torch.no_grad():
        for btok, feats, _gt in dl:
            fg = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else v.dtype, non_blocking=True)
                  for k, v in feats.items()}
            bev = agent(fg)["bev_embed"]
            rows = np.array([P.row_of[t] for t in btok])
            b = P.get(rows, dev)
            delta, _, _ = ck_forward(ck, head, bev, b)
            r = (b["s0"] + delta.double()).cpu().numpy()
            sel.append(LW.select(r, b["allowed"].cpu().numpy()))
            rows_done.append(rows)
    rows = np.concatenate(rows_done)
    idx = np.concatenate(sel)
    base = LW.select(P.s0[rows], P.allowed[rows])
    lab = P.lab_lat[rows]
    ar = np.arange(len(rows))
    out = {"n": int(len(rows)), "pdms": float(lab[ar, idx, P_PDMS].mean()),
           "pdms_ck2": float(lab[ar, base, P_PDMS].mean()), "frac_changed": float(np.mean(idx != base))}
    for k in ("nc", "dac", "ep", "ttc"):
        out[k] = float(lab[ar, idx, LBL[k]].mean())
    return out, rows, idx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run", default="ck2e2e_bevkd")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", default="navtrain_train")
    ap.add_argument("--val", default="navtrain_val")
    ap.add_argument("--test", default="")
    ap.add_argument("--test-only", action="store_true")
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0")
    torch.manual_seed(a.seed)
    run_dir = Path(a.run_dir)
    root = run_dir / "eval" / "root"
    agent, ck, ckpt = build(run_dir, dev)
    head = LW.ListwiseHead().to(dev)
    va = Pool(root, a.run, a.val, need_pool=True)
    if a.limit_val:
        va.rows = va.rows[:a.limit_val]
    if a.test_only:
        st = torch.load(out / "ckpt_best.pt", map_location=dev)
        ck.load_state_dict(st["ck"]); head.load_state_dict(st["head"])
        te = Pool(root, a.run, a.test, need_pool=True)
        ev, rows, idx = run_epoch_eval(agent, ck, head, te, a.test, dev, 8, a.workers)
        np.save(out / f"idx_{a.test}.npy", idx); np.save(out / f"rows_{a.test}.npy", rows)
        res = json.loads((out / "result.json").read_text())
        res["test"] = ev | {"epoch": st["epoch"]}
        (out / "result.json").write_text(json.dumps(res, indent=1))
        log(f"TEST {a.test} (epoch {st['epoch']}): PDMS {ev['pdms']:.4f} (CK2 {ev['pdms_ck2']:.4f}) changed "
            f"{ev['frac_changed']:.3f} n {ev['n']}")
        return 0
    tr = Pool(root, a.run, a.train, need_pool=False)
    tr_rows = tr.rows[:a.limit_train] if a.limit_train else tr.rows
    tr_toks = tr.tokens.token.astype(str).to_numpy()[tr_rows]
    params = [{"params": [p for p in ck.parameters() if p.requires_grad], "lr": a.lr},
              {"params": head.parameters(), "lr": a.lr_head}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    n_steps = a.epochs * (len(tr_toks) // (a.bs * a.accum))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr, a.lr_head], total_steps=max(n_steps, 1), pct_start=0.05)
    ev0, _, _ = run_epoch_eval(agent, ck, head, va, a.val, dev, 8, a.workers)
    log(f"epoch 0: val PDMS {ev0['pdms']:.4f} (CK2 {ev0['pdms_ck2']:.4f}, changed {ev0['frac_changed']:.4f}) n {ev0['n']}")
    hist, best = [], (-1.0, -1)
    step = 0
    for ep in range(1, a.epochs + 1):
        ck.train(); head.train()
        dl = loader_for(a.train, tr_toks, agent, a.bs, a.workers, shuffle_seed=a.seed + ep)
        t0, acc, mb = time.time(), {"exp": [], "lw": [], "kd": []}, 0
        opt.zero_grad(set_to_none=True)
        for btok, feats, _gt in dl:
            fg = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else v.dtype, non_blocking=True)
                  for k, v in feats.items()}
            with torch.no_grad():
                bev = agent(fg)["bev_embed"].detach()
            rows = np.array([tr.row_of[t] for t in btok])
            b = tr.get(rows, dev)
            delta, lg, lg_lat = ck_forward(ck, head, bev, b)
            L = LW.listwise_loss(b["s0"].float() + delta, b["pdms"], b["allowed"], a.tau)
            m = b["allowed"][..., None].float()
            kd = (F.binary_cross_entropy_with_logits(lg, torch.sigmoid(b["logit0"]), reduction="none") * m).sum() / (
                m.sum() * 5) + (F.binary_cross_entropy_with_logits(lg_lat, torch.sigmoid(b["logit0_lat"]),
                                                                  reduction="none") * m).sum() / (m.sum() * 5)
            loss = (L["loss"] + a.kd * kd) / a.accum
            if not torch.isfinite(loss):
                log("non-finite loss: micro-batch skipped")
                continue
            loss.backward()
            acc["exp"].append(float(L["exp_pdms"])); acc["lw"].append(float(L["loss"])); acc["kd"].append(float(kd))
            mb += 1
            if mb % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(list(ck.parameters()) + list(head.parameters()), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                if step < n_steps - 1:
                    sched.step()
                step += 1
                if step % 100 == 0:
                    el = time.time() - t0
                    log(f"ep {ep} step {step}/{n_steps} | E[PDMS] {np.mean(acc['exp'][-800:]):.4f} kd "
                        f"{np.mean(acc['kd'][-800:]):.4f} | {mb * a.bs / el:.1f} tok/s")
        ev, _, _ = run_epoch_eval(agent, ck, head, va, a.val, dev, 8, a.workers)
        hist.append(ev | {"epoch": ep, "train_exp_pdms": float(np.mean(acc["exp"])), "kd": float(np.mean(acc["kd"])),
                          "sec": round(time.time() - t0)})
        log(f"epoch {ep}: val PDMS {ev['pdms']:.4f} (CK2 {ev['pdms_ck2']:.4f}, changed {ev['frac_changed']:.3f}) "
            f"NC {ev['nc']:.4f} DAC {ev['dac']:.4f} EP {ev['ep']:.4f} TTC {ev['ttc']:.4f}")
        torch.save({"ck": ck.state_dict(), "head": head.state_dict(), "epoch": ep, "args": vars(a)}, out / "ckpt_last.pt")
        if ev["pdms"] > best[0]:
            best = (ev["pdms"], ep)
            os.replace(out / "ckpt_last.pt", out / "ckpt_best.pt")
        (out / "result.json").write_text(json.dumps({"args": vars(a), "ckpt": str(ckpt), "val_epoch0": ev0,
                                                     "best_epoch": best[1], "val_best": best[0], "hist": hist},
                                                    indent=1))
    log(f"best val epoch {best[1]}: {best[0]:.4f} (CK2 {ev0['pdms_ck2']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
