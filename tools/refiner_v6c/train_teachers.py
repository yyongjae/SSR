"""Stage T of refiner v6c (report 124 §3, revised for corridor-cf-distill): train the cross-fitted teachers.

  T-D  CorridorHead(t=49) on the z-scored BEVFusion 64 m BEV (kd_bev_0) -> 49 occupancy planes, cf corridor_loss's
       occupancy BCE (pos_weight 4) on the GT corridor64 planes
  T-M  CorridorHead(t=0) on the z-scored ReSMap front64 BEV (kd_bev_1) -> drivable SDF, corridor_loss's SDF term
  T-E  RiskNet (object channels) on the BEVFusion BEV samples + the OOF T-D / T-M fields along a path mixture, labels =
       refiner_v6c.labels.exact_step_masks on the same paths (exact init), as the student's L_E_GT
Every module has one copy per fold (cf_fold = crc32(log) % 2): copy f learns from fold-f tokens only.  T-E copy f
reads the fields of the OTHER fold's T-D / T-M copies on its fold-f tokens, so its inputs are out-of-fold too.
Platt maps are fitted on out-of-fold outputs at the end (occupancy per plane time bin, E_T overlap / approach per time
bin, the T-M off-road probability on the min corner SDF).

No camera images: a token-only dataset feeds the existing target builders (Corridor64TargetBuilder(plan=True),
DistillTeacherTargetBuilder, ExactFullTargetBuilder) through a minimal scene stub.

Usage (GPU runs need approval, report 124 §4-3; GPUs 0-3 only):
  CUDA_VISIBLE_DEVICES=0 python tools/refiner_v6c/train_teachers.py --out <dir> --phase all [--epochs 10]
  smoke on CPU: --device cpu --limit 8 --epochs 1 --batch 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from navsim.agents.para_ssr.modules.corridor_head import Corridor64TargetBuilder  # noqa: E402
from navsim.agents.para_ssr.modules.exact_score import ExactFullTargetBuilder, ego_init  # noqa: E402
from navsim.agents.para_ssr.distill.teacher_store import DistillTeacherTargetBuilder  # noqa: E402
from navsim.agents.para_ssr.modules.torch_tracker import track  # noqa: E402
from navsim.agents.para_ssr.refiner_v6c import labels as LB  # noqa: E402
from navsim.agents.para_ssr.refiner_v6c import losses as LS  # noqa: E402
from navsim.agents.para_ssr.refiner_v6c import nets as N  # noqa: E402
from navsim.agents.para_ssr.refiner_v6c.teacher import TIME_BINS, TeacherSet  # noqa: E402
from navsim.agents.para_ssr.refiner_v6c.mixture import path_mixture  # noqa: E402


# ------------------------------------------------------------------------------------------------- data
class _Scene:
    """the attributes the target builders read: frames[h-1].token, scene_metadata.{num_history_frames, log_name}"""

    def __init__(self, token: str, log: str):
        self.frames = [SimpleNamespace(token=token)]
        self.scene_metadata = SimpleNamespace(num_history_frames=1, log_name=log)


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, cfg, folds=(0, 1), limit: int = 0):
        import pandas as pd
        import zlib
        df = pd.read_parquet(REPO / "resources/km_corridor/packed" / split / "tokens.parquet")
        fold = np.array([zlib.crc32(str(lg).encode()) % 2 for lg in df.log])
        keep = np.isin(fold, folds)
        self.toks = df.token.astype(str).to_numpy()[keep]
        self.logs = df.log.astype(str).to_numpy()[keep]
        if limit:
            self.toks, self.logs = self.toks[:limit], self.logs[:limit]
        self.b_corr = Corridor64TargetBuilder(cfg, plan=True)
        self.b_kd = DistillTeacherTargetBuilder(cfg)
        self.b_ex = ExactFullTargetBuilder(cfg)

    def __len__(self):
        return len(self.toks)

    def __getitem__(self, i):
        sc = _Scene(self.toks[i], self.logs[i])
        out = {}
        for b in (self.b_corr, self.b_kd, self.b_ex):
            out.update(b.compute_targets(sc))
        out.pop("corridor_occ", None)                         # the occ_dist / occ planes below are enough
        return out


def builder_cfg(km_root: str, teacher_root: str, planes: int = 49):
    return SimpleNamespace(corridor_dir=f"{km_root}/corridor64", corridor_tokens_root=str(REPO / "resources/km_corridor/packed"),
                           corridor_occ_planes=planes, exact_dir=f"{km_root}/exact", exact2_dir=f"{km_root}/exact2",
                           distill_feature_root=teacher_root, distill_cache_size=(100, 100), embed_dims=256,
                           bev_h=100, bev_w=100)


def to_dev(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


# ------------------------------------------------------------------------------------------------- paths (T-E)
# ------------------------------------------------------------------------------------------------- phases
def occ_bce(logits, gt, pw=4.0):
    return F.binary_cross_entropy_with_logits(logits, gt, pos_weight=torch.tensor(pw, device=logits.device))


def sdf_loss(pred, gt, valid, bw=4.0):
    w = 1.0 + bw * torch.exp(-gt.abs())
    return ((w * F.smooth_l1_loss(pred, gt, beta=0.5, reduction="none")).mean((1, 2)) * valid).sum() / valid.sum().clamp(min=1)


def train_fields(ts: TeacherSet, data_by_fold, args, dev, log):
    """T-D and T-M, copy f on fold f"""
    for f in range(2):
        params = list(ts.td[f].parameters()) + list(ts.tm[f].parameters())
        for p in params:
            p.requires_grad_(True)
        ts.td[f].train()
        ts.tm[f].train()
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
        dl = torch.utils.data.DataLoader(data_by_fold[f], batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                         drop_last=True, persistent_workers=False)
        step = 0
        for ep in range(args.epochs):
            for bt in dl:
                bt = to_dev(bt, dev)
                ok = bt["kd_ok_0"].bool() & bt["kd_ok_1"].bool() & (bt["corridor_valid"] > 0.5)
                if not bool(ok.any()):
                    continue
                zd = ts.norm_det(bt["kd_bev_0"][ok]).flatten(2).transpose(1, 2)
                zm = ts.norm_map(bt["kd_bev_1"][ok]).flatten(2).transpose(1, 2)
                occ_gt = (bt["corridor_occ_dist"][ok] < 0).float()                         # [b, 49, 200, 200]
                lo = occ_bce(ts.td[f](zd)["corridor_occ"].float(), occ_gt)
                ls = sdf_loss(ts.tm[f](zm)["corridor_sdf"].float(), bt["corridor_sdf"][ok].float(),
                              torch.ones(int(ok.sum()), device=dev))
                loss = lo + ls
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                opt.step()
                step += 1
                if step % args.log_every == 0:
                    log({"phase": "fields", "fold": f, "epoch": ep, "step": step, "occ": float(lo), "sdf": float(ls)})
        ts.td[f].eval()
        ts.tm[f].eval()
        for p in params:
            p.requires_grad_(False)


@torch.no_grad()
def fit_platt_occ(ts: TeacherSet, data_by_fold, args, dev, log, n_px: int = 4096):
    """OOF: copy f on fold 1 - f tokens; logistic regression of GT occupancy on the logit, per time bin"""
    xs = [[] for _ in range(len(TIME_BINS) - 1)]
    ys = [[] for _ in range(len(TIME_BINS) - 1)]
    for f in range(2):
        dl = torch.utils.data.DataLoader(data_by_fold[1 - f], batch_size=args.batch, num_workers=args.workers)
        for n, bt in enumerate(dl):
            if n >= args.platt_batches:
                break
            bt = to_dev(bt, dev)
            ok = bt["kd_ok_0"].bool() & (bt["corridor_valid"] > 0.5)
            if not bool(ok.any()):
                continue
            lg = ts.td[f](ts.norm_det(bt["kd_bev_0"][ok]).flatten(2).transpose(1, 2))["corridor_occ"].float()
            gt = (bt["corridor_occ_dist"][ok] < 0).float()
            planes = torch.arange(1, lg.shape[1] + 1, device=dev)
            kb = ts.platt_occ.bins(planes)
            for kk in range(len(xs)):
                sel = (kb == kk).nonzero(as_tuple=True)[0]
                lv, gv = lg[:, sel].reshape(-1), gt[:, sel].reshape(-1)
                pos = gv.nonzero(as_tuple=True)[0]
                neg = (gv == 0).nonzero(as_tuple=True)[0]
                take = torch.cat([pos[torch.randperm(len(pos), device=dev)[:n_px]],
                                  neg[torch.randperm(len(neg), device=dev)[:n_px]]])
                xs[kk].append(lv[take])
                ys[kk].append(gv[take])
    _fit(ts.platt_occ, xs, ys, log, "occ")


def _fit(platt, xs, ys, log, name, iters: int = 300, min_each: int = 50):
    for kk in range(len(xs)):
        if not xs[kk]:
            continue
        x, y = torch.cat(xs[kk]).float().cpu(), torch.cat(ys[kk]).float().cpu()
        npos, nneg = int((y > 0.5).sum()), int((y <= 0.5).sum())
        if npos < min_each or nneg < min_each:                     # too few of one class: keep the previous map
            log({"phase": "platt", "map": name, "bin": kk, "skipped": True, "npos": npos, "nneg": nneg})
            continue
        a = torch.tensor(float(platt.a[kk]), requires_grad=True)
        bb = torch.tensor(float(platt.b[kk]), requires_grad=True)
        opt = torch.optim.LBFGS([a, bb], max_iter=iters)

        def closure():
            opt.zero_grad()
            l = F.binary_cross_entropy_with_logits(a * x + bb, y)
            l.backward()
            return l

        opt.step(closure)
        platt.a[kk] = float(a.detach())
        platt.b[kk] = float(bb.detach())
        log({"phase": "platt", "map": name, "bin": kk, "a": float(a), "b": float(bb), "n": int(len(x))})


def train_te(ts: TeacherSet, data_by_fold, anchors, args, dev, log):
    """T-E copy f on fold f tokens, OOF fields from the other fold's T-D / T-M, paths = path_mixture"""
    from navsim.agents.para_ssr.refiner_v6c.hook import _area_flags
    gen = torch.Generator().manual_seed(0)
    for f in range(2):
        te = ts.te[f]
        for p in te.parameters():
            p.requires_grad_(True)
        te.train()
        opt = torch.optim.AdamW(te.parameters(), lr=args.lr, weight_decay=1e-4)
        dl = torch.utils.data.DataLoader(data_by_fold[f], batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                         drop_last=True)
        step = 0
        for ep in range(args.epochs):
            for bt in dl:
                bt = to_dev(bt, dev)
                ok = bt["kd_ok_0"].bool() & bt["kd_ok_1"].bool() & (bt["ex1_valid"] > 0.5)
                if not bool(ok.any()):
                    continue
                bt = {k: (v[ok] if torch.is_tensor(v) and v.shape[:1] == ok.shape else v) for k, v in bt.items()}
                b = int(ok.sum())
                with torch.no_grad():
                    copy = torch.full((b,), 1 - f, device=dev, dtype=torch.long)       # OOF fields for fold-f tokens
                    flds = ts.fields(bt["kd_bev_0"], bt["kd_bev_1"], copy)
                    paths = path_mixture(anchors, bt["ex2_hum"].float().cpu(), bt["ex2_hvalid"].cpu(), gen).to(dev)
                    r = paths.shape[1]
                    init_x = ego_init(bt["ex_ego"].double())
                    st9 = track(paths.reshape(b * r, 8, 3).double(), init_x.repeat_interleave(r, 0), full=True,
                                angular=True).reshape(b, r, 41, 9).float()
                    lab = LB.stride2(LB.exact_step_masks(st9[..., :7], bt))
                    status = bt["ex_ego"][:, :4].float()                                # the student's status init
                    S = track(paths.reshape(b * r, 8, 3).float(),
                              torch.cat([status, status.new_zeros(b, 3)], -1).repeat_interleave(r, 0),
                              full=True).reshape(b, r, 41, 7).float()
                    area = _area_flags(S, bt)
                    reads = N.path_reads(S, flds["tok_det"], flds["sdf_m"], flds["occ_prob"], 100, 100, 64.0)
                E = te(reads, area=area)["E"]
                loss, lg = LS.risk_loss(E, lab, None, None, 4.0)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(te.parameters(), 10.0)
                opt.step()
                step += 1
                if step % args.log_every == 0:
                    log({"phase": "te", "fold": f, "epoch": ep, "step": step, "loss": float(loss),
                         **{k: float(v) for k, v in lg.items()}})
        te.eval()
        for p in te.parameters():
            p.requires_grad_(False)


@torch.no_grad()
def fit_platt_te_off(ts: TeacherSet, data_by_fold, anchors, args, dev, log):
    """OOF: T-E copy f (and T-D / T-M copy f) on fold 1 - f tokens and the same path mixture -> Platt of the overlap /
    approach logits per time bin, and of the T-M off-road probability on the min corner SDF (only where all four
    corners are inside 0 <= x <= 64 m, |y| <= 32 m: report 124 D2 g^M)"""
    from navsim.agents.para_ssr.refiner_v6c.hook import _area_flags
    nb = len(TIME_BINS) - 1
    acc = {nm: ([[] for _ in range(nb)], [[] for _ in range(nb)]) for nm in ("ovl", "ttc", "off")}
    gen = torch.Generator().manual_seed(1)
    t21 = N.T21.to(dev)
    kb = ts.platt_ovl.bins(t21)
    for f in range(2):
        dl = torch.utils.data.DataLoader(data_by_fold[1 - f], batch_size=args.batch, num_workers=args.workers)
        for n, bt in enumerate(dl):
            if n >= args.platt_batches:
                break
            bt = to_dev(bt, dev)
            ok = bt["kd_ok_0"].bool() & bt["kd_ok_1"].bool() & (bt["ex1_valid"] > 0.5)
            if not bool(ok.any()):
                continue
            bt = {k: (v[ok] if torch.is_tensor(v) and v.shape[:1] == ok.shape else v) for k, v in bt.items()}
            b = int(ok.sum())
            copy = torch.full((b,), f, device=dev, dtype=torch.long)
            flds = ts.fields(bt["kd_bev_0"], bt["kd_bev_1"], copy)
            paths = path_mixture(anchors, bt["ex2_hum"].float().cpu(), bt["ex2_hvalid"].cpu(), gen).to(dev)
            r = paths.shape[1]
            st9 = track(paths.reshape(b * r, 8, 3).double(), ego_init(bt["ex_ego"].double()).repeat_interleave(r, 0),
                        full=True, angular=True).reshape(b, r, 41, 9).float()
            lab = LB.stride2(LB.exact_step_masks(st9[..., :7], bt))
            status = bt["ex_ego"][:, :4].float()
            S = track(paths.reshape(b * r, 8, 3).float(), torch.cat([status, status.new_zeros(b, 3)], -1)
                      .repeat_interleave(r, 0), full=True).reshape(b, r, 41, 7).float()
            area = _area_flags(S, bt)
            reads = N.path_reads(S, flds["tok_det"], flds["sdf_m"], flds["occ_prob"], 100, 100, 64.0)
            E = ts.te[f](reads, area=area)["E"]
            corners = N.footprint5(S[:, :, t21])[..., 1:, :]
            gm = ((corners[..., 0] >= 0) & (corners[..., 0] <= 64.0) & (corners[..., 1].abs() <= 32.0)).all(-1)
            for kk in range(nb):
                sel = kb == kk
                v = lab["valid"][..., sel] > 0
                acc["ovl"][0][kk].append(E[..., sel, 0][v])
                acc["ovl"][1][kk].append(lab["ovl"][..., sel][v])
                acc["ttc"][0][kk].append(E[..., sel, 1][v])
                acc["ttc"][1][kk].append(lab["ttc"][..., sel][v])
                vo = (lab["valid_off"][..., sel] > 0) & gm[..., sel]
                acc["off"][0][kk].append(reads["ro"][..., sel, 0][vo] * 8.0)                # min corner SDF (m)
                acc["off"][1][kk].append(lab["offroad"][..., sel][vo])
    _fit(ts.platt_ovl, *acc["ovl"], log, "te_ovl")
    _fit(ts.platt_ttc, *acc["ttc"], log, "te_ttc")
    _fit(ts.platt_off, *acc["off"], log, "tm_off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", default="all", choices=("all", "fields", "platt_occ", "te"))
    ap.add_argument("--km-root", default=os.environ.get("KM_ROOT", "/workspace/navsim_workspace/corridor_cache"))
    ap.add_argument("--teacher-root", default=os.environ.get("TEACHER_CACHE_ROOT", "/workspace/teacher_cache"))
    ap.add_argument("--split", default="navtrain_train")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--platt-batches", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", default="")
    args = ap.parse_args()
    dev = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.jsonl", "a")

    def log(d):
        d = {"t": time.time(), **d}
        logf.write(json.dumps(d) + "\n")
        logf.flush()
        print(d, flush=True)

    cfg = builder_cfg(args.km_root, args.teacher_root)
    data = {f: TokenDataset(args.split, cfg, folds=(f,), limit=args.limit) for f in (0, 1)}
    log({"phase": "data", "fold0": len(data[0]), "fold1": len(data[1]), "args": vars(args)})
    ts = TeacherSet().to(dev)
    if args.resume:
        ts.load_state_dict(torch.load(args.resume, map_location="cpu")["state_dict"])
    anchors = torch.tensor(np.load(REPO / "resources/km_corridor/trajectory_anchors_256.npy"), dtype=torch.float32)

    def save(tag):
        meta = {"folds": 2, "hid_e": 192, "planes": 49, "phase_done": tag, "norm": "resources/km_corridor/norm_*_100x100.npz",
                "teacher_inputs": "kd_bev_0 BEVFusion (W flip), kd_bev_1 ReSMap front64 (transpose), student grid"}
        torch.save({"state_dict": ts.state_dict(), "meta": meta}, out / f"teachers_{tag}.pt")
        torch.save({"state_dict": ts.state_dict(), "meta": meta}, out / "teachers.pt")

    if args.phase in ("all", "fields"):
        train_fields(ts, data, args, dev, log)
        save("fields")
    if args.phase in ("all", "platt_occ"):
        fit_platt_occ(ts, data, args, dev, log)
        save("platt_occ")
    if args.phase in ("all", "te"):
        train_te(ts, data, anchors, args, dev, log)
        fit_platt_te_off(ts, data, anchors, args, dev, log)
        save("te")
    log({"phase": "done"})


if __name__ == "__main__":
    main()
