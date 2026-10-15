#!/usr/bin/env python
"""GT-free navtest inference of a CK2 teacher (ck2T DET / ck2M MAP): the teacher alone plans from the 256 raw anchors.

  cd /workspace/yongjae/SSR-ck2 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    nice -n 10 /venv/ssr/bin/python tools/ck/e2e2/eval_teacher_navtest_gtfree.py --run ck2T [--limit 50 --sanity]

No GT / label anywhere in the candidate choice (inputs: BEV cache, status, the fixed anchor set, the teacher).
Selection score (GT-free, no v2): sel = ck_final with w_im = 0
    = 0.5 log(NC + eps) + 0.5 log(DAC + eps) + 1.0 log(5 TTC + 2 C + 5 EP + eps)        (select.ck_final, SEL_W minus im)
on the CK probabilities sigmoid(score_logit) (CK_KEYS nc, dac, ep, ttc, comfort).  Forward = the training eval path
(fp16 autocast, trunk.scene once per token, trunk.candidates per candidate set, score head), lateral decode with
z_lon = 0 (lon head removed in CK2) and slope 0 (correct.correct).

  G1  score all 256 raw anchors -> g1_prob [N,256,5], g1_wlat [N,256,6], g1_lat_traj [N,256,8,3] (lateral head applied
      to every anchor), g1_score [N,256]; P256 = argmax sel (first maximum).
  G2  own top-16 anchors by sel (desc, stable) -> variants.make_variants(speeds (-1, -0.5, 0.5), lats (-0.5, 0.5),
      combine 'separate', ext 'straight') -> 96 pool columns c = k * 6 + v (v: id, a-1.0, a-0.5, a+0.5, l-0.5, l+0.5)
      scored with the same scene features -> prob96 / wlat96 / score96; P96 = select2.select96 with v2 terms removed
      (beta 1, im = 1, w_im = 0): invalid lateral variants (S_8 < 3 m) excluded, ties -> identity columns first.
  G3  lateral head on the P96 pick (w_lat of the picked column, z_lon = 0): modes on / on_except_latvar / off.
Final trajectories final_traj [N, 5, 8, 3], FINAL_MODES order:
  P256 (raw anchor), P256_lat (its lateral correction), P96_off, P96_on, P96_onexl (on_except_latvar).
Official scoring (tools/ck/data/label_cands.py, rows = packed/navtest/tokens.parquet):
  g2_var80_traj [N, 80, 8, 3] (the 5 non-identity variants of the own top-16, column k * 5 + v - 1; identities = raw
  anchors -> labels/navtest/raw256) and final_traj [N, 5, 8, 3].
Outputs (--out, default CK_DATA/ck2/eval/navtest/<run>/gtfree): the arrays above + g2_top16 [N,16] (anchor ids),
  g2_valid96 [N,96], p256_idx [N], p96_col [N], bev_ok / done [N], tokens.parquet, meta.json (ckpt sha, settings,
  timings, counts, sanity), log in KST.  Resumable (done flags).
"""
from __future__ import annotations

import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[3])   # repo root of this worktree
if sys.path[0] != CK:
    sys.path.insert(0, CK)
import navsim  # noqa: E402

assert navsim.__file__.startswith(CK), navsim.__file__

import argparse  # noqa: E402
import hashlib  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from typing import Dict, Optional  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from navsim.agents.para_ssr.ck import constants as Cn  # noqa: E402
from navsim.agents.para_ssr.ck import select2 as S2  # noqa: E402
from navsim.agents.para_ssr.ck.correct import correct  # noqa: E402
from navsim.agents.para_ssr.ck.select import ck_final  # noqa: E402
from navsim.agents.para_ssr.ck.variants import make_variants  # noqa: E402
from navsim.agents.para_ssr.refiner.e2e import ego_inputs  # noqa: E402
from tools.ck import ckutil as U  # noqa: E402

CK_DATA = Path("/home/external-user/ssd/yongjae_refiner/ck")
TRAIN_ROOT = CK_DATA / "ck2" / "train"
EVAL_ROOT = CK_DATA / "ck2" / "eval" / "navtest"
SPLIT = "navtest"
BEV_SHAPE = (256, 50, 100)
W_NOIM = (0.0,) + tuple(Cn.SEL_W[1:])            # SEL_W minus the v2 imitation term
SPEEDS, LATS, COMBINE, EXT = (-1.0, -0.5, 0.5), (-0.5, 0.5), "separate", "straight"
K256, K16, NV = 256, 16, 6
FINAL_MODES = ("P256", "P256_lat", "P96_off", "P96_on", "P96_onexl")
KST = 9 * 3600


def kst(t: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S KST", time.gmtime((time.time() if t is None else t) + KST))


def log(msg: str, fh=None) -> None:
    line = f"[{kst()}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def sel_score(prob: np.ndarray) -> np.ndarray:
    """GT-free selection score (ck_final without im) [.., K, 5] -> [.., K] float64; non-finite -> -inf."""
    p = np.asarray(prob, np.float64)
    s = np.asarray(ck_final(p, np.ones(p.shape[:-1]), W_NOIM), np.float64)
    return np.where(np.isfinite(s), s, -np.inf)


# ----------------------------------------------------------------------------------------------- data
class BevRows(torch.utils.data.Dataset):
    def __init__(self, arm: str, tokens, status: np.ndarray, rows: np.ndarray):
        self.arm, self.tokens, self.status, self.rows = arm, list(tokens), status, np.asarray(rows, np.int64)
        self._cache = None

    def cache(self):
        if self._cache is None:
            if self.arm == "T":
                from navsim.agents.para_ssr.refiner.data import TeacherCache
                self._cache = TeacherCache.for_subset(SPLIT)
            else:
                from navsim.agents.para_ssr.refiner.resmap_cache import ResmapCache
                self._cache = ResmapCache.for_subset(SPLIT)
        return self._cache

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = int(self.rows[i])
        tok = self.tokens[r]
        try:
            x = np.ascontiguousarray(self.cache().load_bev(tok, s_grid=True), dtype=np.float16)
            if x.shape != BEV_SHAPE:
                raise ValueError(f"{tok}: bev {x.shape}")
            ok = True
        except (FileNotFoundError, KeyError, ValueError, OSError):
            x, ok = np.zeros(BEV_SHAPE, np.float16), False
        return {"row": r, "bev": torch.from_numpy(x), "bev_ok": ok,
                "status": torch.from_numpy(np.asarray(self.status[r], np.float32))}


def collate(items):
    return {"row": torch.as_tensor([it["row"] for it in items], dtype=torch.int64),
            "bev": torch.stack([it["bev"] for it in items]),
            "bev_ok": torch.as_tensor([it["bev_ok"] for it in items], dtype=torch.bool),
            "status": torch.stack([it["status"] for it in items])}


# ----------------------------------------------------------------------------------------------- model passes
def score_set(net, feat, mem, cand: torch.Tensor, ego, chunk: int, use_amp: bool) -> Dict[str, torch.Tensor]:
    """CK score head + raw controls of a candidate set [T, K, 8, 3] with fixed scene features (chunks over K;
    candidates are independent: per-candidate self-attention, cross-attention only to the token's global memory)."""
    v0, a0, eds, cmd = ego
    T, K = cand.shape[:2]
    kc = max(1, chunk // max(T, 1))
    outs = {"logit": [], "w_lat": [], "z_lon": []}
    for k0 in range(0, K, kc):
        c = cand[:, k0:k0 + kc].float()
        with torch.autocast(device_type=cand.device.type, dtype=torch.float16, enabled=use_amp):
            o = net.trunk.candidates(feat, mem, c, v0, a0, eds, cmd)
            lg = net.score(o, T, c.shape[1])
        outs["logit"].append(lg.float())
        outs["w_lat"].append(o["w_lat"].float())
        outs["z_lon"].append(o["z_lon"].float())
    return {k: torch.cat(v, 1) for k, v in outs.items()}


def lat_decode(traj: torch.Tensor, w_lat: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
    """decode(traj, z_lon = 0, w_lat), eval slope 0 (= train_ck2.lateral_decode) -> [T, K, 8, 3]."""
    w = w_lat.float()
    return correct(traj.float(), torch.zeros_like(w), w, v0, Cn.LON_ST_SLOPE["eval"])["traj"]


@torch.no_grad()
def infer_batch(net, bev, status, anchors: torch.Tensor, chunk: int, use_amp: bool) -> Dict[str, np.ndarray]:
    dev = bev.device
    T = bev.shape[0]
    ego = ego_inputs(status)
    v0 = ego[0]
    with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
        feat, mem = net.trunk.scene(bev, T)
    # ---- G1: all 256 raw anchors
    A = anchors[None].expand(T, *anchors.shape).contiguous()
    g1 = score_set(net, feat, mem, A, ego, chunk, use_amp)
    g1_prob = torch.sigmoid(g1["logit"]).cpu().numpy()
    g1_lat = lat_decode(A, g1["w_lat"], v0)
    g1_sc = sel_score(g1_prob)
    order = np.argsort(-g1_sc, axis=1, kind="stable")             # desc, ties -> lower anchor id first
    top16 = order[:, :K16].astype(np.int64)
    p256 = top16[:, 0].copy()
    assert (p256 == np.argmax(g1_sc, 1)).all()
    # ---- G2: own top-16 -> 6 variants each -> 96 pool
    tau16 = A[torch.arange(T, device=dev)[:, None], torch.as_tensor(top16, device=dev)]          # [T, 16, 8, 3]
    v0n = torch.nan_to_num(torch.hypot(status[:, 4].float(), status[:, 5].float()), nan=0.0)
    var = make_variants(tau16, v0n, speeds=SPEEDS, lats=LATS, combine=COMBINE, ext=EXT)
    traj96 = var["traj"].reshape(T, K16 * NV, 8, 3)
    valid96 = var["valid"].reshape(T, K16 * NV).cpu().numpy()
    g2 = score_set(net, feat, mem, traj96, ego, chunk, use_amp)
    prob96 = torch.sigmoid(g2["logit"]).cpu().numpy()
    sc96 = sel_score(prob96)
    col = S2.select96(np.zeros((T, K16)), np.ones((T, K16)), prob96, valid96, beta=1.0,
                      types=S2.VARIANT_SETS["all"], w=W_NOIM)
    # ---- G3: lateral head on the P96 pick
    ar = torch.arange(T, device=dev)
    ct = torch.as_tensor(col, device=dev)
    pick = traj96[ar, ct]                                                   # [T, 8, 3]
    pick_lat = lat_decode(pick[:, None], g2["w_lat"][ar, ct][:, None], v0)[:, 0]
    is_latvar = torch.as_tensor(np.isin(col % NV, S2.LAT_TYPES), device=dev)
    pt = torch.as_tensor(p256, device=dev)
    final = torch.stack([A[ar, pt], g1_lat[ar, pt], pick, pick_lat,
                         torch.where(is_latvar[:, None, None], pick, pick_lat)], 1)      # FINAL_MODES order
    # ---- checks (no GT)
    id_traj_eq = bool(torch.equal(traj96[:, ::NV], tau16))
    id_logit_diff = float((g2["logit"][:, ::NV] - g1["logit"][ar[:, None], torch.as_tensor(top16, device=dev)])
                          .abs().max())
    sel_diff_id = float(np.abs(sc96[:, ::NV] - np.take_along_axis(g1_sc, top16, 1)).max())
    return dict(g1_prob=g1_prob.astype(np.float32), g1_wlat=g1["w_lat"].cpu().numpy(),
                g1_lat_traj=g1_lat.cpu().numpy().astype(np.float32), g1_score=g1_sc.astype(np.float32),
                p256_idx=p256, g2_top16=top16,
                g2_var80_traj=var["traj"][:, :, 1:].reshape(T, K16 * (NV - 1), 8, 3).cpu().numpy(),
                g2_valid96=valid96, g2_prob96=prob96.astype(np.float32), g2_wlat96=g2["w_lat"].cpu().numpy(),
                g2_score96=sc96.astype(np.float32), p96_col=col.astype(np.int64),
                final_traj=final.cpu().numpy().astype(np.float32),
                _chk=dict(zlon_absmax=float(max(g1["z_lon"].abs().max(), g2["z_lon"].abs().max())),
                          id_traj_eq=id_traj_eq, id_logit_absdiff=id_logit_diff, id_sel_absdiff=sel_diff_id,
                          n_invalid96=int((~valid96).sum()),
                          lat_absmax_g1=float((g1_lat - A).abs().max()),
                          n_p96_beats_p256=int((np.take_along_axis(sc96, col[:, None], 1)[:, 0]
                                                > np.take_along_axis(g1_sc, p256[:, None], 1)[:, 0] + 1e-9).sum())))


def specs(N: int) -> Dict[str, tuple]:
    return {"g1_prob": ((N, K256, 5), np.float32, np.nan), "g1_wlat": ((N, K256, 6), np.float32, np.nan),
            "g1_lat_traj": ((N, K256, 8, 3), np.float32, np.nan), "g1_score": ((N, K256), np.float32, np.nan),
            "p256_idx": ((N,), np.int64, -1), "g2_top16": ((N, K16), np.int64, -1),
            "g2_var80_traj": ((N, K16 * (NV - 1), 8, 3), np.float32, np.nan),
            "g2_valid96": ((N, K16 * NV), np.bool_, False), "g2_prob96": ((N, K16 * NV, 5), np.float32, np.nan),
            "g2_wlat96": ((N, K16 * NV, 6), np.float32, np.nan), "g2_score96": ((N, K16 * NV), np.float32, np.nan),
            "p96_col": ((N,), np.int64, -1), "final_traj": ((N, len(FINAL_MODES), 8, 3), np.float32, np.nan),
            "bev_ok": ((N,), np.bool_, False), "done": ((N,), np.bool_, False)}


# ----------------------------------------------------------------------------------------------- sanity extras
@torch.no_grad()
def sanity_extra(net, b, anchors, chunk, use_amp) -> Dict[str, float]:
    """Reference checks on one batch: (a) the split scene/candidates path == CKNet.forward (decode False, as the
    training eval) on the first 32 anchors; (b) chunked vs unchunked G1 logits."""
    bev, status = b["bev"], b["status"]
    T = bev.shape[0]
    A = anchors[None].expand(T, *anchors.shape).contiguous()
    with torch.autocast(device_type=bev.device.type, dtype=torch.float16, enabled=use_amp):
        ref = net(bev, A[:, :32], status, decode=False)
        feat, mem = net.trunk.scene(bev, T)
    ego = ego_inputs(status)
    mine = score_set(net, feat, mem, A[:, :32], ego, 10 ** 9, use_amp)
    small = score_set(net, feat, mem, A, ego, 64, use_amp)
    big = score_set(net, feat, mem, A, ego, 10 ** 9, use_amp)
    return {"fwd_vs_split_logit_absdiff": float((ref["score_logit"].float() - mine["logit"]).abs().max()),
            "fwd_vs_split_wlat_absdiff": float((ref["w_lat"].float() - mine["w_lat"]).abs().max()),
            "chunk64_vs_full_logit_absdiff": float((small["logit"] - big["logit"]).abs().max()),
            "chunk64_vs_full_argmax_agree": float((sel_score(torch.sigmoid(small["logit"]).cpu().numpy()).argmax(1)
                                                   == sel_score(torch.sigmoid(big["logit"]).cpu().numpy()).argmax(1))
                                                  .mean())}


# ----------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="ck2T | ck2M | run dir")
    ap.add_argument("--which", default="last")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0, help="first N packed rows only (sanity)")
    ap.add_argument("--sanity", action="store_true", help="extra reference checks on the first batch")
    ap.add_argument("--batch", type=int, default=8, help="tokens per batch")
    ap.add_argument("--chunk", type=int, default=2048, help="max candidates per trunk.candidates call")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true")
    a = ap.parse_args(argv)

    U.gpu_guard("cuda")
    run = Path(a.run) if "/" in a.run else TRAIN_ROOT / a.run
    out = Path(a.out) if a.out else EVAL_ROOT / run.name / "gtfree"
    if not str(out.resolve()).startswith(str(EVAL_ROOT.resolve())):
        raise SystemExit(f"--out must be under {EVAL_ROOT}")
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / "log.txt", "a")
    torch.set_num_threads(1)
    dev = torch.device("cuda:0")
    use_amp = not a.no_amp
    from navsim.agents.para_ssr.ck.model import load_ck
    t_start = time.time()
    net, cfg = load_ck(run, which=a.which, device=str(dev))
    arm = cfg["arm"]
    lon_w = float(net.trunk.lon_head[-1].weight.detach().abs().max()) + float(net.trunk.lon_head[-1].bias.detach().abs().max())

    pk = CK_DATA / "packed" / SPLIT
    tdf = pd.read_parquet(pk / "tokens.parquet")
    status = np.load(pk / "status.npy")
    pok = np.load(pk / "ok.npy")
    Np = len(tdf)
    N = min(a.limit, Np) if a.limit else Np
    anchors_np = np.load(Cn.ANCHORS).astype(np.float32)
    a_sha = hashlib.sha256(anchors_np.tobytes()).hexdigest()[:16]
    if a_sha != cfg["train_data"]["sampler"]["anchors_sha16"]:
        raise SystemExit(f"anchors sha16 {a_sha} != training {cfg['train_data']['sampler']['anchors_sha16']}")
    anchors = torch.from_numpy(anchors_np).to(dev)

    ck_file = run / f"ckpt_{a.which}.pt"
    with U.FileLock(out / ".lock"):
        M = {k: U.open_memmap(out / f"{k}.npy", s, d, fill=f) for k, (s, d, f) in specs(N).items()}
        if not (out / "tokens.parquet").is_file():
            sub = tdf.iloc[:N].copy().reset_index(drop=True)
            sub.to_parquet(out / "tokens.parquet", index=False)
    meta_p = out / "meta.json"
    meta = U.read_json(meta_p, {}) or {}
    meta.update(script=str(Path(__file__).resolve()), run=str(run), which=a.which, arm=arm, split=SPLIT, n=N,
                n_packed=Np, ckpt=str(ck_file), ckpt_sha16=U.sha256_file(ck_file, 16),
                lon_head_absmax_weight_plus_bias=lon_w, anchors=str(Cn.ANCHORS), anchors_sha16=a_sha,
                selection="ck_final without im: 0.5 log NC + 0.5 log DAC + 1.0 log(5 TTC + 2 C + 5 EP), eps 1e-6",
                sel_w=W_NOIM, variants=dict(speeds=SPEEDS, lats=LATS, combine=COMBINE, ext=EXT,
                                            names=list(S2.VNAMES), compute_dtype="float64"),
                tie_rule="P256: first max (lowest anchor id); top16: stable desc; P96: select2.TIE_ORDER "
                         "(identity columns k asc, then variants c asc); invalid lateral variants excluded",
                final_modes=list(FINAL_MODES), lat_decode="correct(z_lon = 0, w_lat, slope 0)",
                amp=use_amp, batch=a.batch, chunk=a.chunk, gt_used="none (no GT / labels read)",
                var80_layout="column k * 5 + (v - 1), v = 1..5 (a-1.0, a-0.5, a+0.5, l-0.5, l+0.5), k = own top-16 rank",
                pool96_layout="column k * 6 + v (select2), identity = g2_top16 anchors")
    U.write_json(meta_p, meta)

    done = M["done"]
    todo = np.flatnonzero(~np.asarray(done[:N], bool) & np.asarray(pok[:N], bool))
    log(f"{run.name} arm {arm} ckpt {ck_file.name} sha16 {meta['ckpt_sha16']} lon_head |w|+|b| max {lon_w:g} -> "
        f"{out}: todo {len(todo)} of {N}", fh)
    if len(todo) == 0:
        fh.close()
        return 0
    ds = BevRows(arm, tdf["token"].astype(str).tolist(), status, todo)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=False, collate_fn=collate,
                                     num_workers=a.workers, pin_memory=True,
                                     prefetch_factor=4 if a.workers > 0 else None)
    chk = {"zlon_absmax": 0.0, "id_traj_eq": True, "id_logit_absdiff": 0.0, "id_sel_absdiff": 0.0,
           "n_invalid96": 0, "lat_absmax_g1": 0.0, "n_p96_beats_p256": 0}
    t0, n, n_bad = time.time(), 0, 0
    sanity = None
    t_gpu, nb = 0.0, 0
    for b in dl:
        nb += 1
        rows = b["row"].numpy()
        bok = b["bev_ok"].numpy()
        n_bad += int((~bok).sum())
        bev = b["bev"].to(dev, non_blocking=True)
        st = b["status"].to(dev).float()
        tg = time.time()
        if a.sanity and sanity is None:
            sanity = sanity_extra(net, {"bev": bev, "status": st}, anchors, a.chunk, use_amp)
            log(f"sanity extra: {sanity}", fh)
        R = infer_batch(net, bev, st, anchors, a.chunk, use_amp)
        t_gpu += time.time() - tg
        c = R.pop("_chk")
        for k in ("zlon_absmax", "id_logit_absdiff", "id_sel_absdiff", "lat_absmax_g1"):
            chk[k] = max(chk[k], c[k])
        chk["id_traj_eq"] = chk["id_traj_eq"] and c["id_traj_eq"]
        chk["n_invalid96"] += c["n_invalid96"]
        chk["n_p96_beats_p256"] += c["n_p96_beats_p256"]
        q = rows[bok]
        for k, v in R.items():
            M[k][q] = v[bok]
        M["bev_ok"][rows] = bok
        done[q] = True
        n += len(rows)
        if nb % 100 == 0 or n == len(todo):
            el = time.time() - t0
            log(f"{n}/{len(todo)} {el:.0f}s {n / max(el, 1e-6):.1f} tok/s (gpu {t_gpu:.0f}s) bev_bad {n_bad} "
                f"zlon {chk['zlon_absmax']:g} id_eq {chk['id_traj_eq']} id_dlogit {chk['id_logit_absdiff']:.2e}", fh)
    for m in M.values():
        m.flush()
    wall = time.time() - t0
    D = np.asarray(M["done"][:N], bool)
    col = np.asarray(M["p96_col"][:N])[D]
    p256 = np.asarray(M["p256_idx"][:N])[D]
    top0 = np.asarray(M["g2_top16"][:N, 0])[D]
    counts = {"n_done": int(D.sum()), "n_bev_bad": int((~np.asarray(M["bev_ok"][:N], bool)).sum()),
              "p96_share_by_variant": {nm: int((col % NV == v).sum()) for v, nm in enumerate(S2.VNAMES)},
              "p96_share_by_rank_k": np.bincount(col // NV, minlength=K16).tolist(),
              "p96_same_as_p256": int((col == 0).sum()),
              "p96_identity_any": int((col % NV == 0).sum()),
              "p96_latvar": int(np.isin(col % NV, S2.LAT_TYPES).sum()),
              "p256_eq_top16_0": bool((p256 == top0).all()),
              "p256_unique_anchors": int(len(np.unique(p256))),
              "valid96_frac": float(np.asarray(M["g2_valid96"][:N])[D].mean()) if D.any() else float("nan")}
    meta = U.read_json(meta_p, {}) or {}
    meta.setdefault("passes", []).append(dict(start=kst(t0), end=kst(), wall_s=round(wall, 1),
                                              gpu_s=round(t_gpu, 1), n=int(n), tok_per_s=round(n / max(wall, 1e-6), 2),
                                              load_s=round(t0 - t_start, 1), checks=chk, sanity=sanity))
    meta["counts"] = counts
    meta["checks"] = chk
    U.write_json(meta_p, meta)
    log(f"done {n} rows in {wall:.0f}s; checks {chk}; counts {counts}", fh)
    fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
