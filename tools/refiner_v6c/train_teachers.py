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
Out-of-fold holds for the heads and the Platt maps only: the kd_bev_0 / kd_bev_1 backbones (BEVFusion 64 m, ReSMap
front64) saw every navtrain token; navtest is held out (report 129 A05, meta 'backbone_in_sample').

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
  label-table coverage (CPU, report 129 A07): --phase check [--min-coverage 1.0] -> <dir>/coverage.json, exit 2 below
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from navsim.agents.para_ssr.modules.corridor_head import SPLITS, Corridor64TargetBuilder  # noqa: E402
from navsim.agents.para_ssr.modules.exact_score import ExactFullTargetBuilder, ego_init  # noqa: E402
from navsim.agents.para_ssr.distill.teacher_store import DistillTeacherTargetBuilder  # noqa: E402
from navsim.agents.para_ssr.modules.torch_tracker import track  # noqa: E402
from navsim.agents.para_ssr.refiner_v6c import labels as LB  # noqa: E402
from navsim.agents.para_ssr.refiner_v6c import losses as LS  # noqa: E402
from navsim.agents.para_ssr.refiner_v6c import nets as N  # noqa: E402
from navsim.agents.para_ssr.refiner_v6c.teacher import READ_RULE, TIME_BINS, TeacherSet  # noqa: E402
from navsim.agents.para_ssr.refiner_v6c.mixture import path_mixture  # noqa: E402


# ------------------------------------------------------------------------------------------------- data
def raster_agents_min1(rows: np.ndarray, planes: int = 49, h: int = 200, w: int = 200, res: float = 0.32) -> np.ndarray:
    """the T-D occupancy target 'raster_min1': corridor_head.raster_agents (same loop, cell convention and cell-centre
    inside test, bitwise) + for a row that marks no cell, the cell holding its corner mean (plane t - 1, if in the grid).
    DESIGN (report 129 A04 / A21): every object occupies at least one cell of the teacher occupancy target, because the
    official NC charges cones and bollards too (~3 % of the objects cover no 0.32 m cell centre).  The base
    corridor_occ / corridor_occ_dist and the student GT guard keep CH.raster_agents."""
    occ = np.zeros((planes, h, w), bool)
    for row in rows.astype(np.float32):
        t = int(row[0])
        if not 1 <= t <= planes:
            continue
        p = row[1:].reshape(4, 2)
        u, v = (-p[:, 1] + 32.0) / res - 0.5, p[:, 0] / res - 0.5                     # col, row in cell units
        j0, j1 = max(int(np.floor(u.min())), 0), min(int(np.ceil(u.max())), w - 1)
        i0, i1 = max(int(np.floor(v.min())), 0), min(int(np.ceil(v.max())), h - 1)
        hit = False
        if j0 <= j1 and i0 <= i1:
            jj, ii = np.meshgrid(np.arange(j0, j1 + 1), np.arange(i0, i1 + 1))
            cr = np.stack([(u[(k + 1) % 4] - u[k]) * (ii - v[k]) - (v[(k + 1) % 4] - v[k]) * (jj - u[k]) for k in range(4)])
            inside = (cr >= 0).all(0) | (cr <= 0).all(0)
            occ[t - 1, i0:i1 + 1, j0:j1 + 1] |= inside
            hit = bool(inside.any())
        if not hit:                                                                     # sub-cell object: its centre cell
            i = int(np.floor(float(p[:, 0].mean()) / res))
            j = int(np.floor((32.0 - float(p[:, 1].mean())) / res))
            if 0 <= i < h and 0 <= j < w:
                occ[t - 1, i, j] = True
    return occ


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
        # city_id = index of the token's city in the sorted cities of the whole split (per-city OOF metrics, F07)
        city = df.city.astype(str).to_numpy() if "city" in df else np.full(len(df), "unknown")
        self.cities = [str(c) for c in np.unique(city)]
        self.city_id = np.searchsorted(np.asarray(self.cities), city)[keep]
        if offset:
            self.toks, self.logs, self.city_id = self.toks[offset:], self.logs[offset:], self.city_id[offset:]
        if limit:
            self.toks, self.logs, self.city_id = self.toks[:limit], self.logs[:limit], self.city_id[:limit]
        self.bev, self.exact = tuple(bev), bool(exact)
        # corr: 'full' = Corridor64TargetBuilder(plan=True) as the student sees it; 'occ' = only the 49 GT occupancy
        # planes as uint8, the T-D target raster_min1 (T-D, its Platt; the base cell-centre raster + one cell per
        # sub-cell object, report 129 A04, so no longer the base occupancy; no distance transforms, no float32 planes);
        # 'sdf' = only the static SDF (T-M); None = no corridor targets (T-E, its Platt).  Each sample crosses the
        # loader's shared memory, whose page faults dominated the cost (report 128 §5)
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
            occ = raster_agents_min1(np.asarray(A["agent_rows"][A["agent_off"][r]:A["agent_off"][r + 1]]))   # [49, 200, 200]
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
        out["city_id"] = torch.tensor(int(self.city_id[i]), dtype=torch.long)
        return out


def builder_cfg(km_root: str, teacher_root: str, planes: int = 49):
    # exact2: $EXACT2_DIR, else <km_root>/exact2 = train.sh's ${EXACT2_DIR:-$KM_ROOT/exact2} (report 129 F02)
    exact2 = os.environ.get("EXACT2_DIR") or f"{km_root}/exact2"
    return SimpleNamespace(corridor_dir=f"{km_root}/corridor64", corridor_tokens_root=str(REPO / "resources/km_corridor/packed"),
                           corridor_occ_planes=planes, exact_dir=f"{km_root}/exact", exact2_dir=exact2,
                           distill_feature_root=teacher_root, distill_cache_size=(100, 100), embed_dims=256,
                           bev_h=100, bev_w=100)


def to_dev(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


# ------------------------------------------------------------------------------------------------- paths (T-E)
# ------------------------------------------------------------------------------------------------- phases
def occ_target(bt) -> torch.Tensor:
    """GT occupancy [B, 49, 200, 200] float: corridor_occ_u8 of corr='occ' (raster_min1) is the T-D target (report 129
    A04).  The occ_dist < 0 fallback is the base cell-centre raster (CH.raster_agents); no teacher phase uses it any more"""
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


def _load_epoch_ckpt(ckpt: Path) -> dict:
    """a per-epoch copy checkpoint to resume; one of older code (no / another read rule) is refused (report 129 F01)"""
    st = torch.load(ckpt, map_location="cpu")
    if st.get("read_rule") != READ_RULE:
        raise RuntimeError(f"{ckpt}: read_rule={st.get('read_rule')!r}, this code needs {READ_RULE!r} (an epoch "
                           f"checkpoint of older code): use a new --out")
    return st


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
        st = _load_epoch_ckpt(ckpt)
        mod.load_state_dict(st["module"])
        opt.load_state_dict(st["opt"])
        ep0, step = int(st["epoch"]) + 1, int(st["step"])
        log({"phase": kind, "fold": f, "resumed_from_epoch": ep0 - 1, "step": step})
    for ep in range(ep0, args.epochs):
        dl = _loader(ds, args, args.batch, seed=1000 * ep + f)
        t0, tot, n = time.time(), 0.0, 0
        n_ok, n_all = 0, 0                                     # rows with a teacher BEV and a GT table (report 129 A07)
        for bt in dl:
            bt = to_dev(bt, dev)
            if kind == "td":
                ok = bt["kd_ok_0"].bool() & (bt["corridor_valid"] > 0.5)
                n_ok, n_all = n_ok + int(ok.sum()), n_all + int(ok.numel())
                if not bool(ok.any()):
                    continue
                z = ts.norm_det(bt["kd_bev_0"][ok]).flatten(2).transpose(1, 2)
                loss = occ_bce(mod(z)["corridor_occ"].float(), occ_target(bt)[ok])
            else:
                ok = bt["kd_ok_1"].bool() & (bt["corridor_valid"] > 0.5)
                n_ok, n_all = n_ok + int(ok.sum()), n_all + int(ok.numel())
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
             "ok_frac": n_ok / max(n_all, 1), "sec": round(time.time() - t0, 1)})
        if ckpt is not None:
            _atomic_save({"module": mod.state_dict(), "opt": opt.state_dict(), "epoch": ep, "step": step,
                          "read_rule": READ_RULE}, ckpt)
    mod.eval()
    for p in params:
        p.requires_grad_(False)


def train_fields(ts: TeacherSet, data, args, dev, log):
    """T-D and T-M, copy f on fold f, one copy after the other (the sequential '--phase all / fields' path).
    data {(kind, f): dataset}: T-D on corr='occ' (raster_min1), T-M on corr='sdf', as the parallel td / tm phases"""
    for f in range(2):
        for kind in ("td", "tm"):
            train_field_copy(ts, kind, f, data[(kind, f)], args, dev, log)


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
    # OOF quality of T-D (report 129 F07; logging only, after the fit): weighted as the fit, before = sigmoid(logit)
    for kk in range(nb):
        if not xs[kk]:
            continue
        x, y, w = (torch.cat(v[kk]).cpu() for v in (xs, ys, ws))
        za = float(ts.platt_occ.a[kk]) * x.double() + float(ts.platt_occ.b[kk])
        log({"phase": "platt_eval", "map": "occ", "bin": kk, "fitted": bool(done[kk]), **_cal_metrics(x, za, y, w)})


def _auc(s: torch.Tensor, y: torch.Tensor):
    """rank AUC (Mann-Whitney, ties at their mean rank) of score s for labels y > 0.5; None without both classes"""
    s, pos = s.double().cpu().numpy(), y.cpu().numpy() > 0.5
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    _, inv, cnt = np.unique(s[order], return_inverse=True, return_counts=True)
    rank = np.empty(len(s))
    rank[order] = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]                                # 1-based mean rank
    return float((rank[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def _cal_metrics(zb: torch.Tensor, za: torch.Tensor, y: torch.Tensor, w=None) -> dict:
    """float64 OOF calibration metrics of one sample: zb / za logits before / after the Platt map, y in {0, 1},
    w weights (None = 1) -> weighted BCE, Brier and mean p before / after, the weighted positive rate, n and the
    unweighted rank AUC of zb (a Platt map with a > 0 keeps it)"""
    zb, za, y = zb.double().cpu(), za.double().cpu(), y.double().cpu()
    w = torch.ones_like(y) if w is None else w.double().cpu()
    sw = w.sum().clamp(min=1e-12)
    out = {"n": int(len(y)), "rate": float((w * y).sum() / sw)}
    for tag, z in (("before", zb), ("after", za)):
        p = torch.sigmoid(z)
        out[f"bce_{tag}"] = float((w * F.binary_cross_entropy_with_logits(z, y, reduction="none")).sum() / sw)
        out[f"brier_{tag}"] = float((w * (p - y) ** 2).sum() / sw)
        out[f"p_{tag}"] = float((w * p).sum() / sw)
    out["auc"] = _auc(zb, y)
    return out


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


def _mask_rows(lab: dict, ok: torch.Tensor) -> dict:
    """rows without a teacher BEV or an exact table: every validity mask of the step labels present ('valid',
    'valid_off', labels.stride2's 'valid_bin') times ok [B]; the step axis is kept"""
    for k in ("valid", "valid_off", "valid_bin"):
        if k in lab:
            lab[k] = lab[k] * ok[:, None, None]
    return lab


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
        st = _load_epoch_ckpt(ckpt)
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
        n_ok, n_all = 0, 0
        for bt in dl:
            bt = to_dev(bt, dev)
            b = int(bt["ex_ego"].shape[0])
            ok = (bt["kd_ok_0"].bool() & bt["kd_ok_1"].bool() & (bt["ex1_valid"] > 0.5)).float()
            n_ok, n_all = n_ok + int(ok.sum()), n_all + b
            with torch.no_grad():
                copy = torch.full((b,), 1 - f, device=dev, dtype=torch.long)           # OOF fields for fold-f tokens
                flds = ts.fields(bt["kd_bev_0"], bt["kd_bev_1"], copy)
                paths = path_mixture(anchors, bt["ex2_hum"].float().cpu(), bt["ex2_hvalid"].cpu(), gen).to(dev)
                r = paths.shape[1]
                init_x = ego_init(bt["ex_ego"].double())
                st9 = track(paths.reshape(b * r, 8, 3).double(), init_x.repeat_interleave(r, 0), full=True,
                            angular=True).reshape(b, r, 41, 9).float()
                lab = _mask_rows(LB.stride2(LB.exact_step_masks(st9[..., :7], bt)), ok)  # mask rows, keep the step
                status = bt["ex_ego"][:, :4].float()                                    # the student's status init
                S = track(paths.reshape(b * r, 8, 3).float(),
                          torch.cat([status, status.new_zeros(b, 3)], -1).repeat_interleave(r, 0),
                          full=True).reshape(b, r, 41, 7).float()
                area = _area_flags(S, bt)
                reads = N.path_reads(S, flds["tok_det"], flds["sdf_m"], flds["occ_prob"], 100, 100, 64.0)
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
             "ok_frac": n_ok / max(n_all, 1), "sec": round(time.time() - t0, 1)})
        if ckpt is not None and rank == 0:
            _atomic_save({"module": te.state_dict(), "opt": opt.state_dict(), "epoch": ep, "step": step,
                          "read_rule": READ_RULE}, ckpt)
        if world > 1:
            torch.distributed.barrier()
    te.eval()
    for p in te.parameters():
        p.requires_grad_(False)


def train_te(ts: TeacherSet, data_by_fold, anchors, args, dev, log):
    """both T-E copies one after the other (the sequential '--phase all / te' path)"""
    for f in range(2):
        train_te_copy(ts, f, data_by_fold[f], anchors, args, dev, log)


def _platt_gates(lab: dict, S: torch.Tensor) -> dict:
    """the steps each T-E / T-M Platt map is fitted on = the steps its KD gate passes (TeacherSet.kd_windows, report
    129 N-te-platt-gate), bool [B, R, T]: ovl valid_bin & inwin, ttc valid_bin & inproj, off valid_off & inwin;
    cont (the gap / vapp / px / py metrics) valid & inproj.  valid_bin = 'valid' without labels.stride2's pooled one"""
    vb = lab.get("valid_bin", lab["valid"]) > 0
    inwin, inproj = TeacherSet.kd_windows(S)
    return {"ovl": vb & inwin, "ttc": vb & inproj, "off": (lab["valid_off"] > 0) & inwin,
            "cont": (lab["valid"] > 0) & inproj}


@torch.no_grad()
def fit_platt_te_off(ts: TeacherSet, data_by_fold, anchors, args, dev, log):
    """OOF: T-E copy f (and T-D / T-M copy f) on fold 1 - f tokens and the same path mixture -> Platt of the overlap /
    approach logits per time bin, and of the T-M off-road probability on the min corner SDF, each on the steps its
    KD gate passes (TeacherSet.kd_windows, report 129 N-te-platt-gate): overlap and off-road where all four corners are
    inside 0 <= x <= 64 m, |y| <= 32 m (report 124 D2 g^M), approach also with the 0.9 s projected box inside.
    Then OOF metrics per bin and per city (logging only, report 129 F07)."""
    from navsim.agents.para_ssr.refiner_v6c.hook import _area_flags
    nb = len(TIME_BINS) - 1
    acc = {nm: ([[] for _ in range(nb)], [[] for _ in range(nb)]) for nm in ("ovl", "ttc", "off")}
    cty = {nm: [[] for _ in range(nb)] for nm in acc}                                   # city_id per sample (F07)
    mae = [torch.zeros(4, dtype=torch.float64) for _ in range(nb)]                       # |E - lab| sums gap/vapp/px/py
    n_mae = [0] * nb
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
            reads = N.path_reads(S, flds["tok_det"], flds["sdf_m"], flds["occ_prob"], 100, 100, 64.0)
            E = ts.te[f](reads, area=area)["E"]
            gates = _platt_gates(lab, S)
            cid = bt["city_id"][:, None, None].expand_as(gates["ovl"])
            y_c = torch.stack([lab["gap"], lab["vapp"], lab["pos"][..., 0], lab["pos"][..., 1]], -1)
            for kk in range(nb):
                sel = kb == kk
                masks = {nm: g[..., sel] for nm, g in gates.items() if nm != "cont"}
                acc["ovl"][0][kk].append(E[..., sel, 0][masks["ovl"]])
                acc["ovl"][1][kk].append(lab["ovl"][..., sel][masks["ovl"]])
                acc["ttc"][0][kk].append(E[..., sel, 1][masks["ttc"]])
                acc["ttc"][1][kk].append(lab["ttc"][..., sel][masks["ttc"]])
                acc["off"][0][kk].append(reads["ro"][..., sel, 0][masks["off"]] * 8.0)    # min corner SDF (m)
                acc["off"][1][kk].append(lab["offroad"][..., sel][masks["off"]])
                for nm, m in masks.items():
                    cty[nm][kk].append(cid[..., sel][m])
                vc = gates["cont"][..., sel]
                mae[kk] += (E[..., sel, 2:6][vc] - y_c[..., sel, :][vc]).abs().double().sum(0).cpu()
                n_mae[kk] += int(vc.sum())
    done = {"ovl": _fit(ts.platt_ovl, *acc["ovl"], log, "te_ovl"), "ttc": _fit(ts.platt_ttc, *acc["ttc"], log, "te_ttc"),
            "off": _fit(ts.platt_off, *acc["off"], log, "tm_off")}
    # OOF quality (report 129 F07; logging only, after the fits): per bin, before = the initial map (sigmoid(logit);
    # T-M off-road sigmoid(-2 s)), after = the fitted one; per city on every bin, each step under its bin's map
    cities = getattr(data_by_fold[0], "cities", [])
    log({"phase": "platt_eval", "cities": list(cities)})
    for nm, name, mp, a0 in (("ovl", "te_ovl", ts.platt_ovl, 1.0), ("ttc", "te_ttc", ts.platt_ttc, 1.0),
                             ("off", "tm_off", ts.platt_off, -2.0)):
        zb_all, za_all, y_all, c_all = [], [], [], []
        for kk in range(nb):
            if not acc[nm][0][kk]:
                continue
            x, y = torch.cat(acc[nm][0][kk]).double().cpu(), torch.cat(acc[nm][1][kk]).double().cpu()
            if len(x) == 0:
                continue
            zb, za = a0 * x, float(mp.a[kk]) * x + float(mp.b[kk])
            log({"phase": "platt_eval", "map": name, "bin": kk, "fitted": bool(done[nm][kk]), **_cal_metrics(zb, za, y)})
            for lst, v in ((zb_all, zb), (za_all, za), (y_all, y), (c_all, torch.cat(cty[nm][kk]).cpu())):
                lst.append(v)
        if not y_all:
            continue
        zb, za, y, c = (torch.cat(v) for v in (zb_all, za_all, y_all, c_all))
        for ci, cname in enumerate(cities):
            m = c == ci
            if bool(m.any()):
                log({"phase": "platt_eval", "map": name, "city": cname, **_cal_metrics(zb[m], za[m], y[m])})
    for kk in range(nb):
        if n_mae[kk]:
            log({"phase": "platt_eval", "map": "te_cont", "bin": kk, "n": n_mae[kk],
                 **{f"mae_{c}": float(mae[kk][j] / n_mae[kk]) for j, c in enumerate(("gap", "vapp", "px", "py"))}})


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


# ------------------------------------------------------------------------------------------------- provenance
BACKBONE_IN_SAMPLE = ("kd_bev_0 (BEVFusion 64 m, trained on navsim_infos_train.pkl = navtrain_train + navtrain_val) and "
                      "kd_bev_1 (ReSMap front64, every navtrain log) saw every navtrain token: OOF holds for the T-D / "
                      "T-M / T-E heads and the Platt maps only (report 129 A05)")


def _git_rev() -> str:
    """'git rev-parse --short HEAD' + '-dirty' when tracked files differ from HEAD; 'unknown' on any error.  'git
    diff' (not diff-index) compares the content of stat-dirty files, as v6c_pipeline.sh's code_rev"""
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True,
                             check=True, timeout=60).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=REPO, capture_output=True,
                               timeout=60).returncode != 0
        return rev + ("-dirty" if dirty else "") if rev else "unknown"
    except Exception:
        return "unknown"


def _read_text(path: str):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return None


def _meta_common(args, cfg) -> dict:
    """read rule, code, seed and label-table provenance of every stage-T artifact (report 129 F01 / F02 / F06)"""
    tables = {"corridor64": cfg.corridor_dir, "exact": cfg.exact_dir, "exact2": cfg.exact2_dir}
    return {"read_rule": READ_RULE, "git": _git_rev(), "seed": int(args.seed), "split": args.split,
            "data": {k: os.path.realpath(str(v)) for k, v in (
                ("corridor_dir", cfg.corridor_dir), ("exact_dir", cfg.exact_dir), ("exact2_dir", cfg.exact2_dir),
                ("teacher_root", cfg.distill_feature_root), ("tokens_root", cfg.corridor_tokens_root))},
            "tables_meta": {f"{nm}/{sp}": _read_text(f"{d}/{sp}/meta.txt") for nm, d in tables.items() for sp in SPLITS},
            "backbone_in_sample": BACKBONE_IN_SAMPLE, "occ_target": "raster_min1 (report 129 A04/A21)"}


def _git_lineage(phase: str, git: str, parts: dict, rmeta: dict, log) -> dict:
    """the git revs of the loaded parts and of the resumed checkpoint vs this run's: a mix is logged and recorded as
    meta git_mixed (not an error)"""
    revs = {k: m.get("git") for k, m in parts.items()}
    if rmeta:
        revs["resume"] = rmeta.get("git")
    revs["this"] = git
    mixed = len(set(revs.values())) > 1 or bool(rmeta.get("git_mixed"))
    if mixed:
        log({"phase": phase, "warning": "parts from different git revisions", "git": revs})
    return {"git_mixed": mixed}


def check_coverage(args) -> int:
    """--phase check (CPU only, report 129 A07): per split, the fraction of the tokens every label table holds
    (corridor64, exact v1 / v2 flags, both teacher BEV stores, the pdm_score_256 tokens) -> <out>/coverage.json;
    -> 2 if any fraction is below --min-coverage, else 0.  The builders read builder_cfg (the packed tokens root), so
    their indices stay valid with --check-tokens, which only replaces the token list that is checked."""
    import pandas as pd
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log_check.jsonl", "a")

    def log(d):
        d = {"t": time.time(), **d}
        logf.write(json.dumps(d) + "\n")
        logf.flush()
        print(d, flush=True)

    cfg = builder_cfg(args.km_root, args.teacher_root)
    b_corr = Corridor64TargetBuilder(cfg, plan=True)
    b_corr._open()
    b_ex = ExactFullTargetBuilder(cfg)
    b_ex._open()
    stores = DistillTeacherTargetBuilder(cfg)._open()
    plan = None
    if args.plan_score:
        with open(f"{args.plan_score}.tokens.json") as fh:
            plan = set(str(t) for t in json.load(fh))
    paths = {"corridor64": os.path.realpath(cfg.corridor_dir), "exact": os.path.realpath(cfg.exact_dir),
             "exact2": os.path.realpath(cfg.exact2_dir), "bevfusion": os.path.realpath(stores[0].teacher_root()),
             "resmap": os.path.realpath(stores[1].teacher_root()), "tokens_root": os.path.realpath(cfg.corridor_tokens_root),
             "plan_score": os.path.realpath(f"{args.plan_score}.tokens.json") if args.plan_score else None}
    for sp in SPLITS:                                    # the v2 split the builder reads (<split>/current if present)
        cur = f"{cfg.exact2_dir}/{sp}/current"
        paths[f"exact2/{sp}"] = os.path.realpath(cur if os.path.exists(cur) else f"{cfg.exact2_dir}/{sp}")
    print("label tables:\n" + "\n".join(f"  {k}: {v}" for k, v in paths.items()), flush=True)
    res = {}
    for sp in [x for x in args.check_splits.split(",") if x]:
        src = args.check_tokens or str(REPO / "resources/km_corridor/packed" / sp / "tokens.parquet")
        toks = pd.read_parquet(src).token.astype(str).tolist()
        n = max(len(toks), 1)

        def frac(has):
            return sum(1 for t in toks if has(t)) / n

        res[sp] = {"corridor64": frac(lambda t: t in b_corr._index),
                   "exact1": frac(lambda t: t in b_ex._index and b_ex._index[t][2]),
                   "exact2": frac(lambda t: t in b_ex._index and b_ex._index[t][3]),
                   "bevfusion": frac(stores[0].has_token), "resmap": frac(stores[1].has_token),
                   "plan_score": None if plan is None else frac(lambda t: t in plan)}
        log({"phase": "check", "split": sp, "tokens": len(toks), "token_list": src, **res[sp]})
    with open(out / "coverage.json", "w") as fh:
        json.dump({"splits": res, "paths": paths, "min_coverage": args.min_coverage}, fh, indent=1)
    low = [f"{sp}/{tb} {v:.6f}" for sp, r in res.items() for tb, v in r.items() if v is not None and v < args.min_coverage]
    if low:
        log({"phase": "check", "error": f"coverage below {args.min_coverage}", "low": low})
        print(f"coverage below --min-coverage {args.min_coverage}: " + ", ".join(low), file=sys.stderr, flush=True)
        return 2
    log({"phase": "check", "ok": True, "min_coverage": args.min_coverage})
    return 0


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
    """load one finished module copy into ts (exactly its keys) -> its meta; a part of older code is refused"""
    path = out / (PART[kind].format(f=f) + ".pt")
    ck = torch.load(path, map_location="cpu")
    rr = ck.get("meta", {}).get("read_rule")
    if rr != READ_RULE:
        raise RuntimeError(f"part {path}: read_rule={rr!r}, this code needs {READ_RULE!r} (report 129 F01): "
                           f"retrain it under a new --out")
    pre, keys = _part_keys(ts, kind, f)
    assert set(ck["state_dict"]) == set(keys), f"{kind}{f}: part keys differ from TeacherSet ({pre})"
    missing, unexpected = ts.load_state_dict(ck["state_dict"], strict=False)
    assert not unexpected and not [k for k in keys if k in missing]
    return ck["meta"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", default="all",
                    choices=("all", "fields", "platt_occ", "te", "td", "tm", "merge_fields", "te_fold", "finish",
                             "check"))
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
    ap.add_argument("--seed", type=int, default=0, help="global torch seed of every phase (+ crc32 of phase:fold)")
    ap.add_argument("--min-coverage", type=float, default=1.0, help="check: the lowest table coverage that passes")
    ap.add_argument("--check-splits", default="navtrain_train,navtrain_val")
    ap.add_argument("--check-tokens", default="", help="check: a tokens parquet replacing every split's list (tests)")
    ap.add_argument("--plan-score", default=os.environ.get("PLAN_SCORE", ""), help="check: <path> of pdm_score_256")
    args = ap.parse_args()
    if args.phase == "check":                          # CPU only: before any CUDA call and the TeacherSet
        sys.exit(check_coverage(args))
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
    rmeta = {}
    if args.resume:                                    # a checkpoint of older code (read rule) is refused (F01)
        rmeta = torch.load(args.resume, map_location="cpu").get("meta", {})
        if rmeta.get("read_rule") != READ_RULE:
            raise RuntimeError(f"--resume {args.resume}: read_rule={rmeta.get('read_rule')!r}, this code needs "
                               f"{READ_RULE!r} (report 129 F01): retrain the earlier phases under a new --out")
    # report 129 F06: one seed per (phase, fold); crc32, as Python's hash() is randomised per process.  Seeds CPU and
    # every CUDA device (the TeacherSet init, the T-E init drawn by merge_fields, the Platt pixel draws)
    torch.manual_seed(args.seed * 1000 + zlib.crc32(f"{args.phase}:{args.fold}".encode()) % 1000)
    ts = frozen_teacher_set(dev, args.resume)
    anchors = torch.tensor(np.load(REPO / "resources/km_corridor/trajectory_anchors_256.npy"), dtype=torch.float32)

    # phases already in the restored checkpoint + the ones this run completes (load_teachers needs all three, so a
    # '--phase te' run resumed from a fields-only checkpoint, without platt_occ, is not taken for a complete teacher)
    phases = list(rmeta.get("phases", []))

    def save(tag, extra=None, parts=None):
        if tag not in phases:
            phases.append(tag)
        common = _meta_common(args, cfg)
        meta = {"folds": 2, "hid_e": 192, "planes": 49, "phase_done": tag, "phases": list(phases),
                "norm": "resources/km_corridor/norm_*_100x100.npz", "epochs": args.epochs, "batch": args.batch,
                "limit": args.limit, "teacher_inputs": "kd_bev_0 BEVFusion (W flip), kd_bev_1 ReSMap front64 "
                                                      "(transpose), student grid", **common,
                **_git_lineage(tag, common["git"], parts or {}, rmeta, log), **(extra or {})}
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
        save_part(ts, args.phase, args.fold, out, {**_meta_common(args, cfg), "epochs": args.epochs, "tokens": len(ds),
                                                   "sec": time.time() - t0})
    elif args.phase == "merge_fields":
        metas = {f"{k}{f}": load_part(ts, k, f, out) for k in ("td", "tm") for f in (0, 1)}
        save("fields", {"parts": metas}, parts=metas)
    elif args.phase == "te_fold":
        ds = data(args.fold, corr=None)
        log({"phase": "te", "fold": args.fold, "tokens": len(ds), "world": world})
        train_te_copy(ts, args.fold, ds, anchors, args, dev, log,
                      ckpt=out / (PART["te_fold"].format(f=args.fold) + ".ckpt"), world=world, rank=rank)
        if rank == 0:
            save_part(ts, "te_fold", args.fold, out, {**_meta_common(args, cfg), "epochs": args.epochs,
                                                      "tokens": len(ds), "world": world, "sec": time.time() - t0})
        if world > 1:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
    elif args.phase == "finish":
        metas = {f"te{f}": load_part(ts, "te_fold", f, out) for f in (0, 1)}
        fit_platt_te_off(ts, {f: data(f, corr=None) for f in (0, 1)}, anchors, args, dev, log)
        save("te", {"parts_te": metas}, parts=metas)
    # ---- sequential phases (smoke / single GPU)
    else:
        if args.phase in ("all", "fields"):
            # T-D on the raster_min1 planes, T-M on the SDF, as the parallel td / tm phases (report 129 A21)
            train_fields(ts, {**{("td", f): data(f, bev=(0,), exact=False, corr="occ") for f in (0, 1)},
                              **{("tm", f): data(f, bev=(1,), exact=False, corr="sdf") for f in (0, 1)}}, args, dev, log)
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
