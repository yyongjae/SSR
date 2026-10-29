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

Usage
  sequential, one GPU:   python tools/refiner_v6c/train_teachers.py --out <dir> --phase all [--epochs 10]
  parallel (scripts/km_corridor/v6c_pipeline.sh, report 128): one process per copy
    --phase td --fold f  /  --phase tm --fold f      -> <dir>/part_{td,tm}{f}.pt  (per-epoch .ckpt, resumed)
    --phase merge_fields                             -> teachers_fields.pt
    --phase platt_occ --resume teachers_fields.pt    -> teachers_platt_occ.pt
    torchrun --nproc_per_node 2 ... --phase te_fold --fold f --resume teachers_platt_occ.pt  -> part_te{f}.pt
    --phase finish --resume teachers_platt_occ.pt    -> teachers_te.pt (T-E Platt + T-M off-road Platt; complete)
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
    """token-only dataset over the packed split, fold-filtered.  bev = which teacher BEVs to read (0 BEVFusion,
    1 ReSMap); exact = whether to build the exact v1 / v2 tables (T-E, its Platt).  T-D reads only BEVFusion and T-M
    only ReSMap: the teacher BEV cache read is the dominant cost of stage T (report 124 §3)."""

    def __init__(self, split: str, cfg, folds=(0, 1), limit: int = 0, bev=(0, 1), exact: bool = True,
                 corr: str = "full", offset: int = 0):
        import pandas as pd
        import zlib
        df = pd.read_parquet(REPO / "resources/km_corridor/packed" / split / "tokens.parquet")
        fold = np.array([zlib.crc32(str(lg).encode()) % 2 for lg in df.log])
        keep = np.isin(fold, folds)
        self.toks = df.token.astype(str).to_numpy()[keep]
        self.logs = df.log.astype(str).to_numpy()[keep]
        if offset:
            self.toks, self.logs = self.toks[offset:], self.logs[offset:]
        if limit:
            self.toks, self.logs = self.toks[:limit], self.logs[:limit]
        self.bev, self.exact = tuple(bev), bool(exact)
        # corr: 'full' = Corridor64TargetBuilder(plan=True) as the student sees it; 'occ' = only the 49 GT occupancy
        # planes as uint8 (T-D, its Platt: occ_dist < 0 is exactly the occupancy, so the 98 distance transforms and
        # the float32 planes are skipped); 'sdf' = only the static SDF (T-M); None = no corridor targets (T-E, its
        # Platt).  Each sample crosses the loader's shared memory, whose page faults dominated the cost (report 128 §5)
        assert corr in ("full", "occ", "sdf", None), corr
        self.corr = corr
        self.b_corr = Corridor64TargetBuilder(cfg, plan=True)
        self.b_kd = DistillTeacherTargetBuilder(cfg)
        self.b_ex = ExactFullTargetBuilder(cfg) if self.exact else None

    def __len__(self):
        return len(self.toks)

    def _kd(self, sc) -> dict:
        if self.bev == (0, 1):
            return self.b_kd.compute_targets(sc)
        stores = self.b_kd._open()
        token = str(sc.frames[0].token)
        h, w = self.b_kd._cache_size
        out = {}
        for i in self.bev:
            arr = stores[i].load_one_token(token)
            ok = arr is not None
            out[f"kd_bev_{i}"] = (torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float16)) if ok
                                  else torch.zeros((self.b_kd._c, h, w), dtype=torch.float16))
            out[f"kd_ok_{i}"] = torch.tensor(ok, dtype=torch.bool)
        return out

    def _corr_light(self, sc) -> dict:
        from navsim.agents.para_ssr.modules.corridor_head import raster_agents
        b = self.b_corr
        b._open()
        hit = b._index.get(str(sc.frames[0].token))
        if hit is None:
            out = {"corridor_valid": torch.tensor(0.0)}
            if self.corr == "occ":
                out["corridor_occ_u8"] = torch.zeros((49, 200, 200), dtype=torch.uint8)
            else:
                out["corridor_sdf"] = torch.zeros(200, 200)
            return out
        A, r = b._arr[hit[0]], hit[1]
        out = {"corridor_valid": torch.tensor(1.0)}
        if self.corr == "occ":
            occ = raster_agents(np.asarray(A["agent_rows"][A["agent_off"][r]:A["agent_off"][r + 1]]))   # [49, 200, 200]
            out["corridor_occ_u8"] = torch.from_numpy(np.ascontiguousarray(occ, dtype=np.uint8))
        else:
            out["corridor_sdf"] = torch.from_numpy(np.asarray(A["static_sdf"][r], np.float32))
        return out

    def __getitem__(self, i):
        sc = _Scene(self.toks[i], self.logs[i])
        out = {}
        if self.corr == "full":
            out.update(self.b_corr.compute_targets(sc))
        elif self.corr is not None:
            out.update(self._corr_light(sc))
        out.update(self._kd(sc))
        if self.b_ex is not None:
            out.update(self.b_ex.compute_targets(sc))
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
def occ_target(bt) -> torch.Tensor:
    """GT occupancy [B, 49, 200, 200] float: the uint8 planes of corr='occ', else occ_dist < 0 (identical)"""
    if "corridor_occ_u8" in bt:
        return bt["corridor_occ_u8"].float()
    return (bt["corridor_occ_dist"] < 0).float()


def occ_bce(logits, gt, pw=4.0):
    return F.binary_cross_entropy_with_logits(logits, gt, pos_weight=torch.tensor(pw, device=logits.device))


def sdf_loss(pred, gt, valid, bw=4.0):
    w = 1.0 + bw * torch.exp(-gt.abs())
    return ((w * F.smooth_l1_loss(pred, gt, beta=0.5, reduction="none")).mean((1, 2)) * valid).sum() / valid.sum().clamp(min=1)


def _atomic_save(obj, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _worker_init(_):
    """one thread per loader worker: the corridor targets run OpenCV distance transforms, whose default thread pool is
    every core; with 4 processes x 16 workers that oversubscribes the host (report 128 §5 timing)"""
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def _loader(ds, args, batch, sampler=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=sampler is None, sampler=sampler, generator=g,
                                       num_workers=args.workers, drop_last=True, persistent_workers=False,
                                       pin_memory=True, worker_init_fn=_worker_init if args.workers > 0 else None)


def train_field_copy(ts: TeacherSet, kind: str, f: int, ds, args, dev, log, ckpt: Path = None):
    """T-D (kind 'td': occupancy BCE pos_weight 4 on BEVFusion) or T-M ('tm': SDF loss on ReSMap), copy f on fold-f
    tokens.  One copy per process, so the four copies train in parallel (report 128 pipeline).  ckpt: a per-epoch
    {module, optimizer, epoch, step} checkpoint; an existing one is resumed from the next epoch."""
    mod = (ts.td if kind == "td" else ts.tm)[f]
    params = list(mod.parameters())
    for p in params:
        p.requires_grad_(True)
    mod.train()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    ep0, step = 0, 0
    if ckpt is not None and ckpt.exists():
        st = torch.load(ckpt, map_location="cpu")
        mod.load_state_dict(st["module"])
        opt.load_state_dict(st["opt"])
        ep0, step = int(st["epoch"]) + 1, int(st["step"])
        log({"phase": kind, "fold": f, "resumed_from_epoch": ep0 - 1, "step": step})
    for ep in range(ep0, args.epochs):
        dl = _loader(ds, args, args.batch, seed=1000 * ep + f)
        t0, tot, n = time.time(), 0.0, 0
        for bt in dl:
            bt = to_dev(bt, dev)
            if kind == "td":
                ok = bt["kd_ok_0"].bool() & (bt["corridor_valid"] > 0.5)
                if not bool(ok.any()):
                    continue
                z = ts.norm_det(bt["kd_bev_0"][ok]).flatten(2).transpose(1, 2)
                loss = occ_bce(mod(z)["corridor_occ"].float(), occ_target(bt)[ok])
            else:
                ok = bt["kd_ok_1"].bool() & (bt["corridor_valid"] > 0.5)
                if not bool(ok.any()):
                    continue
                z = ts.norm_map(bt["kd_bev_1"][ok]).flatten(2).transpose(1, 2)
                loss = sdf_loss(mod(z)["corridor_sdf"].float(), bt["corridor_sdf"][ok].float(),
                                torch.ones(int(ok.sum()), device=dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            step += 1
            tot, n = tot + float(loss), n + 1
            if step % args.log_every == 0:
                log({"phase": kind, "fold": f, "epoch": ep, "step": step, "loss": float(loss),
                     "s_per_step": (time.time() - t0) / max(n, 1)})
        log({"phase": kind, "fold": f, "epoch_done": ep, "step": step, "mean_loss": tot / max(n, 1), "steps": n,
             "sec": round(time.time() - t0, 1)})
        if ckpt is not None:
            _atomic_save({"module": mod.state_dict(), "opt": opt.state_dict(), "epoch": ep, "step": step}, ckpt)
    mod.eval()
    for p in params:
        p.requires_grad_(False)


def train_fields(ts: TeacherSet, data_by_fold, args, dev, log):
    """T-D and T-M, copy f on fold f, one copy after the other (the sequential '--phase all / fields' path)"""
    for f in range(2):
        for kind in ("td", "tm"):
            train_field_copy(ts, kind, f, data_by_fold[f], args, dev, log)


PLATT_SEED = 17


def _platt_loader(ds, args, f: int):
    """the Platt fits read the first platt_batches batches of a SHUFFLED loader (fixed seed per copy).  The packed token
    list is sorted by city (Las Vegas first, 62,875 of 85,109), so the unshuffled loader of the 10-28 run fitted every
    Platt map on Las Vegas tokens of 21-25 logs only (10-28 review)"""
    g = torch.Generator().manual_seed(PLATT_SEED + f)
    return torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, generator=g, num_workers=args.workers,
                                       worker_init_fn=_worker_init)


@torch.no_grad()
def fit_platt_occ(ts: TeacherSet, data_by_fold, args, dev, log, n_px: int = 4096):
    """OOF: copy f on fold 1 - f tokens; logistic regression of GT occupancy on the logit, per time bin.
    Per batch and bin up to n_px positive and n_px negative pixels are drawn (occupancy is ~3 %, so a uniform draw would
    hold few positives).  Each drawn pixel is weighted by 1 / its inclusion probability (class count in that batch-bin /
    number drawn), so the weighted fit is the fit on every pixel: the real occupancy rate, and the batch-to-batch
    density differences (1-6 % per batch) are undone too.  A single global prior shift (e7ae16a) only fixed the mean
    and left the slope ~1.2-2x too steep (report 127 §7.2, measured on the GPU smoke teacher)."""
    nb = len(TIME_BINS) - 1
    xs = [[] for _ in range(nb)]
    ys = [[] for _ in range(nb)]
    ws = [[] for _ in range(nb)]
    npos_all = [0.0] * nb
    nall = [0.0] * nb
    for f in range(2):
        dl = _platt_loader(data_by_fold[1 - f], args, f)
        for n, bt in enumerate(dl):
            if n >= args.platt_batches:
                break
            bt = to_dev(bt, dev)
            ok = bt["kd_ok_0"].bool() & (bt["corridor_valid"] > 0.5)
            if not bool(ok.any()):
                continue
            lg = ts.td[f](ts.norm_det(bt["kd_bev_0"][ok]).flatten(2).transpose(1, 2))["corridor_occ"].float()
            gt = occ_target(bt)[ok]
            planes = torch.arange(1, lg.shape[1] + 1, device=dev)
            kb = ts.platt_occ.bins(planes)
            for kk in range(nb):
                sel = (kb == kk).nonzero(as_tuple=True)[0]
                lv, gv = lg[:, sel].reshape(-1), gt[:, sel].reshape(-1)
                npos_all[kk] += float(gv.sum())
                nall[kk] += float(gv.numel())
                pos = gv.nonzero(as_tuple=True)[0]
                neg = (gv == 0).nonzero(as_tuple=True)[0]
                tp = pos[torch.randperm(len(pos), device=dev)[:n_px]]
                tn = neg[torch.randperm(len(neg), device=dev)[:n_px]]
                xs[kk].append(torch.cat([lv[tp], lv[tn]]))
                ys[kk].append(torch.cat([gv[tp], gv[tn]]))
                ws[kk].append(torch.cat([torch.full((len(tp),), len(pos) / max(len(tp), 1), device=dev),
                                         torch.full((len(tn),), len(neg) / max(len(tn), 1), device=dev)]))
    done = _fit(ts.platt_occ, xs, ys, log, "occ", ws=ws)
    for kk in range(nb):
        if not xs[kk] or nall[kk] == 0:
            continue
        y, w = torch.cat(ys[kk]), torch.cat(ws[kk])
        log({"phase": "platt", "map": "occ_rate", "bin": kk, "fitted": bool(done[kk]), "rate": npos_all[kk] / nall[kk],
             "sample_rate": float(y.mean()), "weighted_rate": float((w * y).sum() / w.sum())})


def _fit(platt, xs, ys, log, name, iters: int = 300, min_each: int = 50, ws=None):
    """per bin: (weighted) BCE logistic fit of p = sigmoid(a x + b); a bin with fewer than min_each positives or
    negatives keeps its previous map.  -> list of bools: fitted per bin"""
    done = [False] * len(xs)
    for kk in range(len(xs)):
        if not xs[kk]:
            continue
        x, y = torch.cat(xs[kk]).float().cpu(), torch.cat(ys[kk]).float().cpu()
        w = None if ws is None else torch.cat(ws[kk]).float().cpu()
        npos, nneg = int((y > 0.5).sum()), int((y <= 0.5).sum())
        if npos < min_each or nneg < min_each:                     # too few of one class: keep the previous map
            log({"phase": "platt", "map": name, "bin": kk, "skipped": True, "npos": npos, "nneg": nneg})
            continue
        w = None if w is None else w / w.mean()
        a = torch.tensor(float(platt.a[kk]), requires_grad=True)
        bb = torch.tensor(float(platt.b[kk]), requires_grad=True)
        opt = torch.optim.LBFGS([a, bb], max_iter=iters)

        def closure():
            opt.zero_grad()
            l = F.binary_cross_entropy_with_logits(a * x + bb, y, weight=w)
            l.backward()
            return l

        opt.step(closure)
        platt.a[kk] = float(a.detach())
        platt.b[kk] = float(bb.detach())
        done[kk] = True
        log({"phase": "platt", "map": name, "bin": kk, "a": float(a), "b": float(bb), "n": int(len(x)),
             "weighted": w is not None})
    return done


def train_te_copy(ts: TeacherSet, f: int, ds, anchors, args, dev, log, ckpt: Path = None, world: int = 1,
                  rank: int = 0):
    """T-E copy f on fold-f tokens, OOF fields from the other fold's T-D / T-M, paths = path_mixture.  With world > 1
    (torchrun) the copy trains under DDP with per-rank batch = args.batch / world, so the optimizer sees the same
    batch size as the single-GPU recipe.  Rows without a teacher BEV or an exact table are masked out of the loss
    (not dropped: every rank keeps the same number of steps)."""
    from navsim.agents.para_ssr.refiner_v6c.hook import _area_flags
    te = ts.te[f]
    for p in te.parameters():
        p.requires_grad_(True)
    te.train()
    model = te
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(te, device_ids=[dev.index])
    assert args.batch % world == 0, (args.batch, world)
    opt = torch.optim.AdamW(te.parameters(), lr=args.lr, weight_decay=1e-4)
    ep0, step = 0, 0
    if ckpt is not None and ckpt.exists():
        st = torch.load(ckpt, map_location="cpu")
        te.load_state_dict(st["module"])
        opt.load_state_dict(st["opt"])
        ep0, step = int(st["epoch"]) + 1, int(st["step"])
        log({"phase": "te", "fold": f, "resumed_from_epoch": ep0 - 1, "step": step})
    sampler = None
    if world > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True,
                                                                  seed=f, drop_last=True)
    gen = torch.Generator().manual_seed(1000 * f + rank)
    for ep in range(ep0, args.epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        dl = _loader(ds, args, args.batch // world, sampler=sampler, seed=1000 * ep + f)
        t0, tot, n = time.time(), 0.0, 0
        for bt in dl:
            bt = to_dev(bt, dev)
            b = int(bt["ex_ego"].shape[0])
            ok = (bt["kd_ok_0"].bool() & bt["kd_ok_1"].bool() & (bt["ex1_valid"] > 0.5)).float()
            with torch.no_grad():
                copy = torch.full((b,), 1 - f, device=dev, dtype=torch.long)           # OOF fields for fold-f tokens
                flds = ts.fields(bt["kd_bev_0"], bt["kd_bev_1"], copy)
                paths = path_mixture(anchors, bt["ex2_hum"].float().cpu(), bt["ex2_hvalid"].cpu(), gen).to(dev)
                r = paths.shape[1]
                init_x = ego_init(bt["ex_ego"].double())
                st9 = track(paths.reshape(b * r, 8, 3).double(), init_x.repeat_interleave(r, 0), full=True,
                            angular=True).reshape(b, r, 41, 9).float()
                lab = LB.stride2(LB.exact_step_masks(st9[..., :7], bt))
                for k in ("valid", "valid_off"):                                       # mask rows, keep the step
                    lab[k] = lab[k] * ok[:, None, None]
                status = bt["ex_ego"][:, :4].float()                                    # the student's status init
                S = track(paths.reshape(b * r, 8, 3).float(),
                          torch.cat([status, status.new_zeros(b, 3)], -1).repeat_interleave(r, 0),
                          full=True).reshape(b, r, 41, 7).float()
                area = _area_flags(S, bt)
                reads = N.path_reads(S, flds["tok_det"], flds["sdf_m"], flds["occ_prob"], 100, 100, 64.0, legacy_rear=True)
            E = model(reads, area=area)["E"]
            loss, lg = LS.risk_loss(E, lab, None, None, 4.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(te.parameters(), 10.0)
            opt.step()
            step += 1
            tot, n = tot + float(loss), n + 1
            if step % args.log_every == 0:
                log({"phase": "te", "fold": f, "epoch": ep, "step": step, "loss": float(loss),
                     "s_per_step": (time.time() - t0) / max(n, 1), **{k: float(v) for k, v in lg.items()}})
        log({"phase": "te", "fold": f, "epoch_done": ep, "step": step, "mean_loss": tot / max(n, 1), "steps": n,
             "sec": round(time.time() - t0, 1)})
        if ckpt is not None and rank == 0:
            _atomic_save({"module": te.state_dict(), "opt": opt.state_dict(), "epoch": ep, "step": step}, ckpt)
        if world > 1:
            torch.distributed.barrier()
    te.eval()
    for p in te.parameters():
        p.requires_grad_(False)


def train_te(ts: TeacherSet, data_by_fold, anchors, args, dev, log):
    """both T-E copies one after the other (the sequential '--phase all / te' path)"""
    for f in range(2):
        train_te_copy(ts, f, data_by_fold[f], anchors, args, dev, log)


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
        dl = _platt_loader(data_by_fold[1 - f], args, f)
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
            reads = N.path_reads(S, flds["tok_det"], flds["sdf_m"], flds["occ_prob"], 100, 100, 64.0, legacy_rear=True)
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


def frozen_teacher_set(dev, resume: str = "") -> TeacherSet:
    """TeacherSet (optionally restored), every module frozen and in eval (BatchNorm running stats, no dropout) until a
    phase switches the ones it trains: train_fields -> td[f] / tm[f], train_te -> te[f].  A fresh nn.Module is in
    train mode, so without this a '--phase platt_occ / te --resume' run would read the T-D / T-M fields with batch
    statistics and overwrite their running stats (report 127 §2.2).  TeacherSet.train() always maps to eval."""
    ts = TeacherSet().to(dev)
    if resume:
        ts.load_state_dict(torch.load(resume, map_location="cpu")["state_dict"])
    ts.eval()
    ts.requires_grad_(False)
    return ts


PARALLEL_PHASES = ("td", "tm", "te_fold")          # one module copy (--fold) per process
PART = {"td": "part_td{f}", "tm": "part_tm{f}", "te_fold": "part_te{f}"}


def _part_keys(ts: TeacherSet, kind: str, f: int):
    pre = {"td": "td.", "tm": "tm.", "te_fold": "te.", "te": "te."}[kind] + f"{f}."
    return pre, [k for k in ts.state_dict() if k.startswith(pre)]


def save_part(ts: TeacherSet, kind: str, f: int, out: Path, meta: dict):
    pre, keys = _part_keys(ts, kind, f)
    sd = ts.state_dict()
    _atomic_save({"state_dict": {k: sd[k] for k in keys}, "meta": {"kind": kind, "fold": f, **meta}},
                 out / (PART[kind].format(f=f) + ".pt"))


def load_part(ts: TeacherSet, kind: str, f: int, out: Path) -> dict:
    """load one finished module copy into ts (exactly its keys) -> its meta"""
    ck = torch.load(out / (PART[kind].format(f=f) + ".pt"), map_location="cpu")
    pre, keys = _part_keys(ts, kind, f)
    assert set(ck["state_dict"]) == set(keys), f"{kind}{f}: part keys differ from TeacherSet ({pre})"
    missing, unexpected = ts.load_state_dict(ck["state_dict"], strict=False)
    assert not unexpected and not [k for k in keys if k in missing]
    return ck["meta"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", default="all",
                    choices=("all", "fields", "platt_occ", "te", "td", "tm", "merge_fields", "te_fold", "finish"))
    ap.add_argument("--fold", type=int, default=-1, help="td / tm / te_fold: the copy (= its training fold)")
    ap.add_argument("--km-root", default=os.environ.get("KM_ROOT", "/workspace/navsim_workspace/corridor_cache"))
    ap.add_argument("--teacher-root", default=os.environ.get("TEACHER_CACHE_ROOT", "/workspace/teacher_cache"))
    ap.add_argument("--split", default="navtrain_train")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0, help="skip the first N tokens of each fold (timing probes)")
    ap.add_argument("--platt-batches", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", default="")
    args = ap.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        assert args.phase == "te_fold", "torchrun / DDP is only for --phase te_fold"
        torch.distributed.init_process_group("nccl")
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dev = torch.device("cuda", local)
    else:
        dev = torch.device(args.device)
    if args.phase in PARALLEL_PHASES:
        assert args.fold in (0, 1), "--fold 0|1 is needed for td / tm / te_fold"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.phase + (f"_f{args.fold}" if args.phase in PARALLEL_PHASES else "") + (f"_r{rank}" if rank else "")
    logf = open(out / f"log_{tag}.jsonl", "a")

    def log(d):
        d = {"t": time.time(), "rank": rank, **d}
        logf.write(json.dumps(d) + "\n")
        logf.flush()
        if rank == 0:
            print(d, flush=True)

    cfg = builder_cfg(args.km_root, args.teacher_root)

    def data(f, **kw):
        return TokenDataset(args.split, cfg, folds=(f,), limit=args.limit, offset=args.offset, **kw)

    log({"phase": "start", "args": vars(args), "world": world})
    ts = frozen_teacher_set(dev, args.resume)
    anchors = torch.tensor(np.load(REPO / "resources/km_corridor/trajectory_anchors_256.npy"), dtype=torch.float32)

    # phases already in the restored checkpoint + the ones this run completes (load_teachers needs all three, so a
    # '--phase te' run resumed from a fields-only checkpoint, without platt_occ, is not taken for a complete teacher)
    phases = list(torch.load(args.resume, map_location="cpu").get("meta", {}).get("phases", [])) if args.resume else []

    def save(tag, extra=None):
        if tag not in phases:
            phases.append(tag)
        meta = {"folds": 2, "hid_e": 192, "planes": 49, "phase_done": tag, "phases": list(phases),
                "norm": "resources/km_corridor/norm_*_100x100.npz", "epochs": args.epochs, "batch": args.batch,
                "limit": args.limit, "teacher_inputs": "kd_bev_0 BEVFusion (W flip), kd_bev_1 ReSMap front64 "
                                                      "(transpose), student grid", **(extra or {})}
        _atomic_save({"state_dict": ts.state_dict(), "meta": meta}, out / f"teachers_{tag}.pt")
        _atomic_save({"state_dict": ts.state_dict(), "meta": meta}, out / "teachers.pt")

    t0 = time.time()
    # ---- parallel pipeline (report 128 §5): td / tm copies, merge, platt_occ, te copies (DDP), finish
    if args.phase in ("td", "tm"):
        ds = data(args.fold, bev=(0,) if args.phase == "td" else (1,), exact=False,
                  corr="occ" if args.phase == "td" else "sdf")
        log({"phase": args.phase, "fold": args.fold, "tokens": len(ds)})
        train_field_copy(ts, args.phase, args.fold, ds, args, dev, log,
                         ckpt=out / (PART[args.phase].format(f=args.fold) + ".ckpt"))
        save_part(ts, args.phase, args.fold, out, {"epochs": args.epochs, "tokens": len(ds), "sec": time.time() - t0})
    elif args.phase == "merge_fields":
        metas = {f"{k}{f}": load_part(ts, k, f, out) for k in ("td", "tm") for f in (0, 1)}
        save("fields", {"parts": metas})
    elif args.phase == "te_fold":
        ds = data(args.fold, corr=None)
        log({"phase": "te", "fold": args.fold, "tokens": len(ds), "world": world})
        train_te_copy(ts, args.fold, ds, anchors, args, dev, log,
                      ckpt=out / (PART["te_fold"].format(f=args.fold) + ".ckpt"), world=world, rank=rank)
        if rank == 0:
            save_part(ts, "te_fold", args.fold, out, {"epochs": args.epochs, "tokens": len(ds), "world": world,
                                                      "sec": time.time() - t0})
        if world > 1:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
    elif args.phase == "finish":
        metas = {f"te{f}": load_part(ts, "te_fold", f, out) for f in (0, 1)}
        fit_platt_te_off(ts, {f: data(f, corr=None) for f in (0, 1)}, anchors, args, dev, log)
        save("te", {"parts_te": metas})
    # ---- sequential phases (smoke / single GPU)
    else:
        if args.phase in ("all", "fields"):
            train_fields(ts, {f: data(f, bev=(0, 1), exact=False) for f in (0, 1)}, args, dev, log)
            save("fields")
        if args.phase in ("all", "platt_occ"):
            fit_platt_occ(ts, {f: data(f, bev=(0,), exact=False, corr="occ") for f in (0, 1)}, args, dev, log)
            save("platt_occ")
        if args.phase in ("all", "te"):
            dd = {f: data(f, corr=None) for f in (0, 1)}
            train_te(ts, dd, anchors, args, dev, log)
            fit_platt_te_off(ts, dd, anchors, args, dev, log)
            save("te")
    log({"phase": "done", "sec": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
