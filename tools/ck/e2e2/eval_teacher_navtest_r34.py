#!/usr/bin/env python
"""CK2 teachers (ck2T DET / ck2M MAP, ckpt_last) on the v2 r34 navtest top-16 -- reference A (user 2026-10-08:
"참고용 평가도 2,3에서 같이 진행해").  GT-free selection; GT / official labels only for the metrics (stage eval).

  # 1) variants of the 16 r34 candidates (CPU, teacher independent, shared by both teachers)
  cd /workspace/yongjae/SSR-ck2 && PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
    /venv/ssr/bin/python tools/ck/e2e2/eval_teacher_navtest_r34.py --stage variants
  # 2) teacher inference + selection (one GPU per teacher: GPU 2 = ck2T, GPU 3 = ck2M)
  CUDA_VISIBLE_DEVICES=2 ... eval_teacher_navtest_r34.py --stage infer,select --teacher ck2T
  CUDA_VISIBLE_DEVICES=3 ... eval_teacher_navtest_r34.py --stage infer,select --teacher ck2M
  # 3) official scoring (tools/ck/data/label_cands.py, CPU) of the files listed in <teacher>/r34/select.json
  # 4) metrics (whatever labels exist; rows needing missing labels are listed as pending)
  ... eval_teacher_navtest_r34.py --stage eval --teacher ck2T
  --limit N: the first N packed rows into <out-root>/smoke<N>/ (sanity checks run there: forward equivalence with
  CKNet.forward, z_lon == 0, identity variants bitwise, zero lateral = identity).

Inputs (rows = /home/external-user/ssd/yongjae_refiner/ck/packed/navtest/tokens.parquet, N = 12146)
  cand.npy [N,16,8,3] (v2 r34 top-16 executed, column 0 = v2's submission), status.npy [N,8], v2_final / v2_im [N,16]
  BEV: tools/ck/data/ck_dataset.CKDataset(bev = arm): T = refiner.data.TeacherCache (BEVFusion t0, navtest cache),
  M = refiner.resmap_cache.ResmapCache (navtest cache); BEV z-score norm from the run dir (model.load_ck).
Model: navsim.agents.para_ssr.ck.model.load_ck(<CK_DATA>/ck2/train/<teacher>, 'last') (user: last epoch); the same
  ops as CKNet.forward (trunk.scene once per token, trunk.candidates on the 16 candidates and, with the same scene
  features, on the 80 variants), fp16 autocast as train_ck2 eval; lateral decode = ck/correct.correct(cand,
  z_lon = 0 (forced), w_lat, v0 = |(vx, vy)|, slope 0) for every candidate and variant.
Variants: ck/variants.make_variants(cand16, v0 = hypot(status[4], status[5]) NaN -> 0, speeds (-1, -0.5, 0.5),
  lats (-0.5, 0.5), combine 'separate', ext 'straight', float64 decode) -> traj80 [N,80,8,3], column j = k * 5 + (v - 1),
  v = 1..5 = a-1.0, a-0.5, a+0.5, l-0.5, l+0.5 (select2.VNAMES[1:]); valid80 [N,80] (False = lateral variant of a
  candidate with S_8 < 3 m: a duplicate of its parent, never selectable).  96-pool (select2 layout) column c = k * 6 + v.
Selection (no GT, no labels)
  W_NOIM = SEL_W without w_im: score = 0.5 log NC + 0.5 log DAC + 1.0 log(5 TTC + 2 C + 5 EP + eps)  (CK-only planner)
  r34_ck_noim     argmax_k over the 16 originals of the CK-only score (np.argmax: ties -> lower k)
  r34_old_b1      select.select_all(v2_final, v2_im, prob, beta 1)['a']   (old formula with v2 im)
  r34_old_b0.5    beta 0.5 (context)
  pool96_ck_noim  select2.select96 over the 16 originals + valid variants, CK-only score (ties -> identity columns
                  first, k ascending, then variants in c order = parent first)
  pool96_old_b1   select2.select96 with v2 im, beta 1 (descriptive context)
  each mode: _a = the selected trajectory, _b = its lateral correction (b-style); v2_lat = lateral of column 0.
Outputs <out-root> (default CK_DATA/ck2/eval/navtest)
  r34_variants/  traj80.npy f32 [N,80,8,3], valid80.npy bool [N,80], v0.npy f32 [N], dec.npz (dev_xy, ds4, d_end,
                 alpha, beta, ext_m [N,16,6]), built.npy bool [N], meta.json
  <teacher>/r34/ score_logit16.npy f32 [N,16,5], score_logit80.npy f32 [N,80,5] (CK_KEYS nc dac ep ttc comfort),
                 prob16 / prob80 (sigmoid, written by select), w_lat16 / w_lat80 f32 [N,K,6], e_lat16 / e_lat80 f32
                 [N,K,8], lat_traj16 f32 [N,16,8,3], lat_traj80 f32 [N,80,8,3], done.npy, bev_ok.npy, infer_meta.json,
                 picks.npz (idx per mode; r34 0..15, pool 0..95), picks.parquet, final_traj.npy f32 [N,M,8,3]
                 (every mode, a and b, names in select.json), score_stack.npy f32 [N,S,8,3] (the trajectories that
                 need official labels: b modes + v2_lat), select.json, and after scoring + stage eval:
                 table.md, metrics.json, per_token.parquet.
Official labels used by stage eval: CK_DATA/labels/navtest/cand (r34 top-16), <out-root>/r34_variants/labels
  (label_cands --traj traj80.npy --out ...), <out-root>/<teacher>/r34/labels_stack (label_cands --traj
  score_stack.npy --out ...).
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
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from typing import Dict, List, Optional  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tools.ck import ckutil as U  # noqa: E402

CK_DATA = Path("/home/external-user/ssd/yongjae_refiner/ck")
OUT_ROOT = CK_DATA / "ck2" / "eval" / "navtest"
SPLIT = "navtest"
K16, NV, NVAR = 16, 6, 5
SPEEDS, LATS, COMBINE, EXT = (-1.0, -0.5, 0.5), (-0.5, 0.5), "separate", "straight"
KST = 9 * 3600
R34_MODES = (("r34_ck_noim", "noim", None), ("r34_old_b1", "old", 1.0), ("r34_old_b0.5", "old", 0.5))
POOL_MODES = (("pool96_ck_noim", "noim", 1.0), ("pool96_old_b1", "old", 1.0))
STAGES = ("variants", "infer", "select", "eval")


def kst(t: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S KST", time.gmtime((time.time() if t is None else t) + KST))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def w_noim():
    from navsim.agents.para_ssr.ck.constants import SEL_W
    return (0.0,) + tuple(SEL_W[1:])


def packed():
    d = CK_DATA / "packed" / SPLIT
    tdf = pd.read_parquet(d / "tokens.parquet")
    ok = np.load(d / "ok.npy")
    return d, tdf, ok


def roots(a):
    root = Path(a.out_root) / (f"smoke{a.limit}" if a.limit else "")
    return root, root / "r34_variants", (root / a.teacher / "r34") if a.teacher else None


def todo_rows(a, n: int, ok: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(ok)
    return rows[: a.limit] if a.limit else rows


def save_npy(path: Path, arr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp{os.getpid()}.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------- stage variants
def stage_variants(a) -> Dict:
    import torch

    from navsim.agents.para_ssr.ck import variants as VV
    t0 = time.time()
    root, vd, _ = roots(a)
    pdir, tdf, pok = packed()
    N = len(tdf)
    rows = todo_rows(a, N, pok)
    if (vd / "meta.json").is_file() and U.read_json(vd / "meta.json").get("complete"):
        built = np.load(vd / "built.npy")
        if built[rows].all():
            print(f"[variants] {vd} complete, skipped", flush=True)
            return U.read_json(vd / "meta.json")
    cand = np.load(pdir / "cand.npy", mmap_mode="r")
    st = np.load(pdir / "status.npy", mmap_mode="r")
    traj80 = np.zeros((N, K16 * NVAR, 8, 3), np.float32)
    valid80 = np.zeros((N, K16 * NVAR), bool)
    v0a = np.full(N, np.nan, np.float32)
    built = np.zeros(N, bool)
    dec = {k: np.full((N, K16, NV), np.nan, np.float32) for k in ("dev_xy", "ds4", "d_end", "alpha", "beta", "ext_m")}
    names = None
    n_bad_id = 0
    for i in range(0, len(rows), a.var_chunk):
        rr = rows[i:i + a.var_chunk]
        tau = np.asarray(cand[rr, :K16], np.float32)
        s = np.asarray(st[rr], np.float64)
        v0 = np.nan_to_num(np.hypot(s[:, 4], s[:, 5]), nan=0.0)
        out = VV.make_variants(torch.from_numpy(tau.copy()), torch.from_numpy(v0), SPEEDS, LATS, COMBINE,
                               compute_dtype=torch.float64, ext=EXT)
        tr = out["traj"].numpy()                                         # [T, 16, 6, 8, 3]
        if not np.array_equal(tr[:, :, 0], tau):
            n_bad_id += 1
        names = out["meta"]["names"]
        traj80[rr] = tr[:, :, 1:].reshape(len(rr), K16 * NVAR, 8, 3)
        valid80[rr] = out["valid"].numpy()[:, :, 1:].reshape(len(rr), K16 * NVAR)
        v0a[rr] = v0.astype(np.float32)
        for k in dec:
            dec[k][rr] = out["meta"][k].numpy()
        built[rr] = True
        if (i // a.var_chunk) % 5 == 0:
            print(f"[variants] {i + len(rr)}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    from navsim.agents.para_ssr.ck.select2 import VNAMES
    assert tuple(names) == tuple(VNAMES), (names, VNAMES)
    assert n_bad_id == 0, "identity variant must be the candidate bytes"
    vd.mkdir(parents=True, exist_ok=True)
    save_npy(vd / "traj80.npy", traj80)
    save_npy(vd / "valid80.npy", valid80)
    save_npy(vd / "v0.npy", v0a)
    save_npy(vd / "built.npy", built)
    np.savez(vd / "dec.npz", **dec)
    vv = valid80[rows].reshape(len(rows), K16, NVAR)
    meta = dict(split=SPLIT, n=N, n_built=int(built.sum()), limit=a.limit, source=str(pdir / "cand.npy"),
                cand_sha16=U.sha256_file(pdir / "cand.npy"), speeds=SPEEDS, lats=LATS, combine=COMBINE, ext=EXT,
                s_on_frac=VV.S_ON_FRAC, compute_dtype="float64", version=VV.VERSION, names=list(names),
                column="j = k * 5 + (v - 1), v = 1..5 = names[1:]", v0="hypot(status[4], status[5]) NaN -> 0",
                valid_frac_per_variant={n: float(vv[..., j].mean()) for j, n in enumerate(names[1:])},
                dev_xy_lt_1cm_frac={n: float((dec["dev_xy"][rows][..., j + 1] < 0.01).mean())
                                    for j, n in enumerate(names[1:])},
                sec=round(time.time() - t0, 1), finished=kst(), complete=True)
    U.write_json(vd / "meta.json", meta)
    print(f"[variants] -> {vd} ({meta['sec']} s) valid {meta['valid_frac_per_variant']}", flush=True)
    return meta


# ----------------------------------------------------------------------------------------------- stage infer
def resolve_run(teacher: str) -> Path:
    p = Path(teacher)
    if p.is_dir():
        return p
    q = CK_DATA / "ck2" / "train" / teacher
    if q.is_dir():
        return q
    raise SystemExit(f"teacher run not found: {teacher}")


def teacher_forward(net, bev, cand, status, var):
    """= CKNet.forward(bev, cand, status, extra=var, decode=False) plus the variants' w_lat (same scene features)."""
    from navsim.agents.para_ssr.refiner.e2e import ego_inputs
    T = cand.shape[0]
    ego = ego_inputs(status)
    v0, a0, eds, cmd = ego
    feat, mem = net.trunk.scene(bev, T)
    o16 = net.trunk.candidates(feat, mem, cand.float(), v0, a0, eds, cmd)
    o80 = net.trunk.candidates(feat, mem, var.float(), v0, a0, eds, cmd)
    return {"s16": net.score(o16, T, cand.shape[1]), "s80": net.score(o80, T, var.shape[1]),
            "w16": o16["w_lat"].float(), "w80": o80["w_lat"].float(),
            "z16": o16["z_lon"].float(), "z80": o80["z_lon"].float(), "v0": v0}


def lat_decode(tau, w, v0):
    """decode(tau, z_lon = 0 (forced), w_lat, slope 0) (= train_ck2.lateral_decode at eval)."""
    import torch

    from navsim.agents.para_ssr.ck import constants as Cn
    from navsim.agents.para_ssr.ck.correct import correct
    return correct(tau.float(), torch.zeros_like(w), w.float(), v0, Cn.LON_ST_SLOPE["eval"])


def sanity_checks(net, b, var, dev) -> Dict:
    """Smoke: teacher_forward == CKNet.forward (bitwise), z_lon == 0, zero lateral = identity."""
    import torch
    out = {}
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        mine = teacher_forward(net, b["bev"], b["cand"], b["status"], var)
        ref = net(b["bev"], b["cand"], b["status"], extra=var, decode=False)
    out["fwd_score16_maxdiff"] = float((mine["s16"] - ref["score_logit"]).abs().max())
    out["fwd_score80_maxdiff"] = float((mine["s80"] - ref["extra_score_logit"]).abs().max())
    out["fwd_wlat16_maxdiff"] = float((mine["w16"] - ref["w_lat"]).abs().max())
    out["z_lon_absmax"] = float(max(mine["z16"].abs().max(), mine["z80"].abs().max()))
    lh = net.trunk.lon_head[-1]
    out["lon_head_absmax"] = float(max(lh.weight.abs().max(), lh.bias.abs().max()))
    c0 = lat_decode(b["cand"], torch.zeros_like(mine["w16"]), mine["v0"])
    out["zero_lat_identity_maxdiff"] = float((c0["traj"] - b["cand"].float()).abs().max())
    # K-independence of the per-candidate scores (no cross-candidate attention): first 4 candidates alone
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        r4 = net(b["bev"], b["cand"][:, :4], b["status"], decode=False)
    out["k_subset_score_maxdiff"] = float((r4["score_logit"] - mine["s16"][:, :4]).abs().max())
    return out


def stage_infer(a) -> Dict:
    import torch
    from torch.utils.data import DataLoader

    from navsim.agents.para_ssr.ck.model import load_ck
    from tools.ck.data.ck_dataset import CKDataset, collate_ck

    t0 = time.time()
    root, vd, td = roots(a)
    run = resolve_run(a.teacher)
    if a.device.startswith("cuda"):
        U.gpu_guard(a.device)
        dev = torch.device("cuda:0")
    else:
        dev = torch.device("cpu")
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    net, cfg = load_ck(run, which=a.which, device=str(dev))
    net.eval()
    arm = cfg["arm"]
    pdir, tdf, pok = packed()
    N = len(tdf)
    rows = todo_rows(a, N, pok)
    vmeta = U.read_json(vd / "meta.json") or {}
    built = np.load(vd / "built.npy")
    if not vmeta.get("complete") or not built[rows].all():
        raise SystemExit(f"{vd}: variants not built for every row (run --stage variants first)")
    var_all = np.load(vd / "traj80.npy", mmap_mode="r")
    td.mkdir(parents=True, exist_ok=True)
    K80 = K16 * NVAR
    specs = {"done": ((N,), np.bool_, False), "bev_ok": ((N,), np.bool_, False),
             "score_logit16": ((N, K16, 5), np.float32, np.nan), "score_logit80": ((N, K80, 5), np.float32, np.nan),
             "w_lat16": ((N, K16, 6), np.float32, np.nan), "w_lat80": ((N, K80, 6), np.float32, np.nan),
             "e_lat16": ((N, K16, 8), np.float32, np.nan), "e_lat80": ((N, K80, 8), np.float32, np.nan),
             "lat_traj16": ((N, K16, 8, 3), np.float32, np.nan), "lat_traj80": ((N, K80, 8, 3), np.float32, np.nan)}
    A = {k: U.open_memmap(td / f"{k}.npy", s, d, fill=f) for k, (s, d, f) in specs.items()}
    if not (td / "tokens.parquet").is_file():
        sub = tdf.copy()
        sub["row"] = np.arange(N)
        sub.to_parquet(td / "tokens.parquet", index=False)
    done = A["done"]
    rr = rows[~np.asarray(done[rows], bool)]
    ck_file = run / f"ckpt_{a.which}.pt"
    meta = dict(teacher=a.teacher, run=str(run), which=a.which, arm=arm, ckpt=str(ck_file),
                ckpt_sha16=U.sha256_file(ck_file), cfg_epochs=cfg.get("epochs"), split=SPLIT, n=N, limit=a.limit,
                device=str(dev), cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"), batch=a.batch,
                workers=a.workers, amp="fp16 autocast (as train_ck2 eval)", z_lon="forced 0 in the lateral decode",
                lat_slope=0.0, bev_norm=str(run / ("norm.npz" if arm == "T" else "norm_map.npz")),
                variants=str(vd / "traj80.npy"), started=kst(t0))
    print(f"[infer] {a.teacher} arm {arm} ({ck_file.name}) -> {td}: todo {len(rr)} of {len(rows)}", flush=True)
    sanity = None
    n, n_bad = 0, 0
    if len(rr):
        ds = CKDataset(SPLIT, bev=arm, k=K16, labels=None, gt=False, rows=rr)
        dl = DataLoader(ds, batch_size=a.batch, shuffle=False, collate_fn=collate_ck, num_workers=a.workers,
                        pin_memory=dev.type == "cuda", prefetch_factor=4 if a.workers > 0 else None)
        use_amp = dev.type == "cuda"
        t1 = time.time()
        with torch.no_grad():
            for b in dl:
                prow = b["rows"].numpy().astype(np.int64)
                bok = b["bev_ok"].numpy().astype(bool).reshape(-1)
                n_bad += int((~bok).sum())
                x = {"bev": b["bev"].to(dev, non_blocking=True), "cand": b["cand"].to(dev).float(),
                     "status": b["status"].to(dev).float()}
                var = torch.from_numpy(np.asarray(var_all[prow], np.float32)).to(dev)
                if a.limit and sanity is None:
                    sanity = sanity_checks(net, x, var, dev)
                    print(f"[infer] sanity {sanity}", flush=True)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                    o = teacher_forward(net, x["bev"], x["cand"], x["status"], var)
                c16 = lat_decode(x["cand"], o["w16"], o["v0"])
                c80 = lat_decode(var, o["w80"], o["v0"])
                g = lambda t: t.float().cpu().numpy()[bok]
                q = prow[bok]                         # rows without BEV stay NaN / not done
                A["score_logit16"][q] = g(o["s16"])
                A["score_logit80"][q] = g(o["s80"])
                A["w_lat16"][q] = g(o["w16"])
                A["w_lat80"][q] = g(o["w80"])
                A["e_lat16"][q] = g(c16["e_lat"])
                A["e_lat80"][q] = g(c80["e_lat"])
                A["lat_traj16"][q] = g(c16["traj"])
                A["lat_traj80"][q] = g(c80["traj"])
                A["bev_ok"][prow] = bok
                done[q] = True
                n += len(prow)
                if n % (a.batch * 100) < a.batch or n == len(rr):
                    el = time.time() - t1
                    print(f"[infer] {n}/{len(rr)} {el:.0f}s {n / max(el, 1e-6):.1f} tok/s", flush=True)
        for m in A.values():
            m.flush()
    old = U.read_json(td / "infer_meta.json", {}) or {}
    meta.update(n_rows=int(len(rows)), n_done=int(np.asarray(done[rows]).sum()), n_bev_bad_this_call=n_bad,
                sec=round(time.time() - t0, 1), finished=kst(), sanity=sanity or old.get("sanity"),
                sessions=(old.get("sessions") or []) + [dict(started=meta["started"], finished=kst(), n=n,
                                                             sec=round(time.time() - t0, 1))])
    U.write_json(td / "infer_meta.json", meta)
    print(f"[infer] done {n} rows ({n_bad} without BEV) in {meta['sec']} s", flush=True)
    return meta


# ----------------------------------------------------------------------------------------------- stage select
def to_pool(x16: np.ndarray, x80: np.ndarray) -> np.ndarray:
    """[N,16,...] + [N,80,...] -> [N,96,...] in select2 layout c = k * 6 + v (v = 0 identity)."""
    n = x16.shape[0]
    tail = x16.shape[2:]
    out = np.empty((n, K16, NV) + tail, x16.dtype)
    out[:, :, 0] = x16
    out[:, :, 1:] = x80.reshape((n, K16, NVAR) + tail)
    return out.reshape((n, K16 * NV) + tail)


def select_modes(prob16, prob80, valid80, v2_final, v2_im) -> Dict[str, np.ndarray]:
    """GT-free picks -> {mode: idx [n]} (r34 modes 0..15, pool modes 0..95)."""
    from navsim.agents.para_ssr.ck import select2 as S2
    from navsim.agents.para_ssr.ck.select import ck_final, select_all
    n = prob16.shape[0]
    ones16, zeros16 = np.ones((n, K16)), np.zeros((n, K16))
    picks = {}
    for name, kind, beta in R34_MODES:
        if kind == "noim":
            s = ck_final(prob16.astype(np.float64), ones16, w_noim())
            s = np.where(np.isfinite(s), s, -np.inf)
            picks[name] = np.argmax(s, 1).astype(np.int64)
        else:
            picks[name] = np.asarray(select_all(v2_final, v2_im, prob16, beta=beta)["a"], np.int64)
    prob96 = to_pool(prob16, prob80)
    valid96 = to_pool(np.ones((n, K16), bool), valid80)
    for name, kind, beta in POOL_MODES:
        if kind == "noim":
            picks[name] = S2.select96(zeros16, ones16, prob96, valid96, beta=1.0, w=w_noim())
        else:
            picks[name] = S2.select96(v2_final, v2_im, prob96, valid96, beta=beta)
    return picks


def stage_select(a) -> Dict:
    t0 = time.time()
    root, vd, td = roots(a)
    pdir, tdf, pok = packed()
    N = len(tdf)
    rows = todo_rows(a, N, pok)
    done = np.load(td / "done.npy")
    rows = rows[done[rows]]
    cand = np.load(pdir / "cand.npy", mmap_mode="r")
    v2_final = np.asarray(np.load(pdir / "v2_final.npy", mmap_mode="r")[rows], np.float64)
    v2_im = np.asarray(np.load(pdir / "v2_im.npy", mmap_mode="r")[rows], np.float64)
    L = {k: np.load(td / f"{k}.npy", mmap_mode="r") for k in ("score_logit16", "score_logit80", "lat_traj16",
                                                              "lat_traj80", "e_lat16", "e_lat80")}
    prob16 = sigmoid(L["score_logit16"][rows])
    prob80 = sigmoid(L["score_logit80"][rows])
    valid80 = np.load(vd / "valid80.npy")[rows]
    traj80 = np.load(vd / "traj80.npy", mmap_mode="r")
    picks = select_modes(prob16, prob80, valid80, v2_final, v2_im)
    n = len(rows)
    ar = np.arange(n)
    c16 = np.asarray(cand[rows, :K16], np.float32)
    l16 = np.asarray(L["lat_traj16"][rows], np.float32)
    pool = to_pool(c16, np.asarray(traj80[rows], np.float32))
    lpool = to_pool(l16, np.asarray(L["lat_traj80"][rows], np.float32))
    elat = to_pool(np.asarray(L["e_lat16"][rows]), np.asarray(L["e_lat80"][rows]))
    names, finals, stack_names, stack = [], [], [], []
    info = {}
    for mode, idx in picks.items():
        isr = mode.startswith("r34")
        A_ = (c16 if isr else pool)[ar, idx]
        B_ = (l16 if isr else lpool)[ar, idx]
        names += [f"{mode}_a", f"{mode}_b"]
        finals += [A_, B_]
        stack_names.append(f"{mode}_b")
        stack.append(B_)
        e = (elat[ar, (idx * NV if isr else idx)])[:, 2:]
        d = {"frac_changed": float(np.mean(idx != 0)), "lat_abs_e_mean": float(np.abs(e).mean()),
             "lat_frac_any_0p1m": float((np.abs(e).max(1) > 0.1).mean()),
             "lat_max_dev_m_mean": float(np.linalg.norm(B_[..., :2] - A_[..., :2], axis=-1).max(1).mean())}
        if isr:
            d["pick_hist_k"] = np.bincount(idx, minlength=K16).tolist()
        else:
            from navsim.agents.para_ssr.ck.select2 import VNAMES
            d["sel_share"] = {VNAMES[t]: float(np.mean(idx % NV == t)) for t in range(NV)}
            d["frac_parent_not_v2"] = float(np.mean(idx // NV != 0))
        info[mode] = d
    names.append("v2_lat")
    finals.append(l16[:, 0])
    stack_names.append("v2_lat")
    stack.append(l16[:, 0])
    M, S = len(names), len(stack_names)
    final = np.full((N, M, 8, 3), np.nan, np.float32)
    final[rows] = np.stack(finals, 1)
    sst = np.full((N, S, 8, 3), np.nan, np.float32)
    sst[rows] = np.stack(stack, 1)
    # rows without inference (none expected on the full split): v2's own trajectory so label_cands sees finite poses;
    # stage eval excludes them through done
    miss = np.setdiff1d(np.arange(N), rows)
    if len(miss):
        v2c = np.asarray(cand[miss, 0], np.float32)
        final[miss] = v2c[:, None]
        sst[miss] = v2c[:, None]
    save_npy(td / "final_traj.npy", final)
    save_npy(td / "score_stack.npy", sst)
    P16 =np.full((N, K16, 5), np.nan, np.float32)
    P16[rows] = prob16
    P80 = np.full((N, K16 * NVAR, 5), np.nan, np.float32)
    P80[rows] = prob80
    save_npy(td / "prob16.npy", P16)
    save_npy(td / "prob80.npy", P80)
    pk = {m: np.full(N, -1, np.int64) for m in picks}
    for m, idx in picks.items():
        pk[m][rows] = idx
    np.savez(td / "picks.npz", rows=rows, **pk)
    df = pd.DataFrame({"token": tdf.token.to_numpy()[rows], "row": rows})
    for m, idx in picks.items():
        df[m] = idx.astype(np.int16)
    df.to_parquet(td / "picks.parquet", index=False)
    sel = dict(teacher=a.teacher, n=N, n_rows=int(n), limit=a.limit, final_names=names, stack_names=stack_names,
               score_stack=str(td / "score_stack.npy"), final_traj=str(td / "final_traj.npy"),
               w_noim=w_noim(), modes={m: dict(kind=k, beta=b) for m, k, b in R34_MODES + POOL_MODES},
               per_mode=info, gt_free=("selection reads prob16 / prob80 / valid80 / v2_final / v2_im only; "
                                       "no gt_traj, no labels"),
               scoring=dict(
                   variants=dict(traj=str(vd / "traj80.npy"), out=str(vd / "labels"), name="ck2eval_r34_var80",
                                 shape=[N, K16 * NVAR, 8, 3], note="teacher independent: score once for ck2T + ck2M"),
                   stack=dict(traj=str(td / "score_stack.npy"), out=str(td / "labels_stack"),
                              name=f"ck2eval_{a.teacher}_r34_stack", shape=[N, S, 8, 3])),
               sec=round(time.time() - t0, 1), finished=kst())
    U.write_json(td / "select.json", sel)
    print(json.dumps({m: {k: v for k, v in d.items() if k != "pick_hist_k"} for m, d in info.items()}, indent=1),
          flush=True)
    print(f"[select] -> {td} ({sel['sec']} s)", flush=True)
    return sel


# ----------------------------------------------------------------------------------------------- stage eval
def load_lab(d: Path, n_expect: int, k_expect: int):
    if not (d / "labels.npy").is_file():
        return None, None
    lab = np.load(d / "labels.npy")
    ok = np.load(d / "ok.npy")
    assert lab.shape[0] == n_expect and lab.shape[1] == k_expect, (d, lab.shape, n_expect, k_expect)
    return lab.astype(np.float64), ok.astype(bool)


def stage_eval(a) -> Dict:
    from navsim.agents.para_ssr.ck import constants as Cn
    from navsim.agents.para_ssr.ck.select2 import VNAMES
    from tools.ck.eval_ck import csv_check, fmt_ci, lead_flags, log_bootstrap, row_metrics

    t0 = time.time()
    root, vd, td = roots(a)
    pdir, tdf, pok = packed()
    N = len(tdf)
    sel = U.read_json(td / "select.json")
    pz = np.load(td / "picks.npz")
    rows = pz["rows"]
    labc, okc = load_lab(CK_DATA / "labels" / SPLIT / "cand", N, K16)
    labv, okv = load_lab(Path(a.labels_var) if a.labels_var else vd / "labels", N, K16 * NVAR)
    labs, oks = load_lab(Path(a.labels_stack) if a.labels_stack else td / "labels_stack", N, len(sel["stack_names"]))
    valid80 = np.load(vd / "valid80.npy")
    have_v, have_s = labv is not None, labs is not None
    keep = pok[rows] & okc[rows].all(1)
    if have_v:
        keep &= (okv[rows] | ~valid80[rows]).all(1)
    if have_s:
        keep &= oks[rows].all(1)
    r = rows[keep]
    n = len(r)
    ar = np.arange(n)
    tokens, logs = tdf.token.to_numpy()[r], tdf.log.to_numpy()[r]
    lead_mask, lead_src = lead_flags(SPLIT, tokens)
    P = Cn.LBL["pdms"]
    LC = labc[r]
    LP = to_pool(LC, labv[r]) if have_v else None
    LS = labs[r] if have_s else None
    base = LC[:, 0]
    fb = ((base[:, Cn.LBL["nc"]] < 1) | (base[:, Cn.LBL["ttc"]] < 1)).astype(np.float64)
    out_rows, per_tok, pending = [], [], []

    def add(name, chosen, idx=None, extra=None):
        m = row_metrics(chosen, lead_mask)
        m["mode"] = name
        if idx is not None:
            m["frac_changed"] = float(np.mean(idx != 0))
        if name != "v2":
            m["d_pdms"] = log_bootstrap(chosen[:, P] - base[:, P], logs, a.bootstrap, 0)
            if lead_mask is not None and lead_mask.any():
                fv = ((chosen[:, Cn.LBL["nc"]] < 1) | (chosen[:, Cn.LBL["ttc"]] < 1)).astype(np.float64)
                m["d_lead_fail"] = log_bootstrap((fv - fb)[lead_mask], logs[lead_mask], a.bootstrap, 0)
        if extra:
            m.update(extra)
        out_rows.append(m)
        df = pd.DataFrame({"token": tokens, "log": logs, "mode": name,
                           "chosen": (idx if idx is not None else np.zeros(n, np.int64)).astype(np.int16)})
        for c in Cn.LABEL_COLS:
            df[c] = chosen[:, Cn.LBL[c]].astype(np.float32)
        per_tok.append(df)

    add("v2", base, np.zeros(n, np.int64))
    o16 = LC[..., P].argmax(1)
    add("oracle16", LC[ar, o16], o16)
    if have_v:
        lp = np.where(to_pool(np.ones((n, K16), bool), valid80[r])[..., None], LP, -np.inf)
        o96 = lp[..., P].argmax(1)
        add("oracle96", LP[ar, o96], o96)
    sidx = {nm: j for j, nm in enumerate(sel["stack_names"])}
    for mode in [m for m, _, _ in R34_MODES] + [m for m, _, _ in POOL_MODES]:
        idx = pz[mode][r]
        isr = mode.startswith("r34")
        ex = {}
        if not isr:
            ex["sel_share"] = {VNAMES[t]: float(np.mean(idx % NV == t)) for t in range(NV)}
        if isr:
            add(f"{mode}_a", LC[ar, idx], idx, ex)
        elif have_v:
            add(f"{mode}_a", LP[ar, idx], idx, ex)
        else:
            pending.append(f"{mode}_a (variant labels)")
        if have_s:
            add(f"{mode}_b", LS[:, sidx[f"{mode}_b"]], idx, ex)
        else:
            pending.append(f"{mode}_b (stack labels)")
    if have_s:
        add("v2_lat", LS[:, sidx["v2_lat"]], np.zeros(n, np.int64))
    else:
        pending.append("v2_lat (stack labels)")
    info = dict(teacher=a.teacher, split=SPLIT, n_total=N, n_rows_selected=int(len(rows)), n_eval=n,
                n_logs=int(len(np.unique(logs))), labels_cand=str(CK_DATA / "labels" / SPLIT / "cand"),
                labels_var=str(vd / "labels") if have_v else None, labels_stack=str(td / "labels_stack") if have_s
                else None, pending=pending, lead_src=lead_src,
                lead_n=int(lead_mask.sum()) if lead_mask is not None else None, n_boot=a.bootstrap,
                token_set=("packed ok & inference done & 16 cand labels ok"
                           + (" & valid variant labels ok" if have_v else "") + (" & stack labels ok" if have_s else "")),
                infer_meta=U.read_json(td / "infer_meta.json"), created=kst())
    if not a.limit:
        info["v2_check"] = csv_check(tdf.token.to_numpy(), labc[:, 0, P], pok & okc[:, 0])
    ref = {}
    for old in ("ckT_p1", "ckM_p1"):
        mj = U.read_json(CK_DATA / "eval" / old / SPLIT / "metrics.json")
        if mj:
            ref[old] = {f"{rr['variant']}({rr['beta']:g})" if rr.get("beta") is not None else rr["variant"]:
                        round(100 * rr["pdms"], 2) for rr in mj["rows"] if rr["variant"] in ("v2", "a", "b")}
    info["old_ck_navtest_pdms"] = ref
    metrics = dict(info, rows=out_rows)
    U.write_json(td / "metrics.json", metrics)
    pd.concat(per_tok, ignore_index=True).to_parquet(td / "per_token.parquet", index=False)
    Lm = [f"# CK2 teacher {a.teacher} (ckpt_last) / navtest / reference A (v2 r34 top-16)", "",
          f"토큰 {n} / 전체 {N} ({info['token_set']}). log {info['n_logs']}. bootstrap {a.bootstrap}회 (log 군집). "
          "PDMS 0-100. 후보 선택은 GT/라벨 없이 (CK 확률, v2 im/final만); 라벨은 지표 계산에만 사용.", "",
          "모드: _a = 선택한 궤적 그대로, _b = 선택 궤적에 lateral head 교정 (z_lon=0). ck_noim = 0.5 log NC + 0.5 log DAC "
          "+ log(5 TTC + 2 C + 5 EP) (v2 없음). old_b1 / old_b0.5 = 이전 식 (v2 im 포함, β). pool96 = r34 16개 + "
          "변형 80개 (a-1.0, a-0.5, a+0.5 straight, l±0.5; 무효 lateral 제외, 동점은 부모 우선).", ""]
    if info.get("v2_check"):
        c = info["v2_check"]
        Lm += [f"v2 점검: 평균 pdms {100 * c['v2_mean_pdms']:.2f} vs CSV {100 * c['csv_pdms']:.3f} "
               f"(n {c['n']}), v2_mismatch={c['v2_mismatch']}", ""]
    Lm += ["| 모드 | PDMS | ΔPDMS vs v2 [95% CI] | NC | DAC | EP | TTC | C | NC 실패 | DAC 실패 | TTC 실패 | "
           "앞차 감속 NC/TTC 실패 (n) | Δ앞차 실패 [95% CI] | 바뀐 비율 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in out_rows:
        lead = (f"{100 * m['lead_fail_nc_ttc']:.2f} ({m['lead_n']})" if m.get("lead_n") else "-")
        Lm.append(f"| {m['mode']} | {100 * m['pdms']:.2f} | {fmt_ci(m.get('d_pdms'))} | {100 * m['nc']:.2f} | "
                  f"{100 * m['dac']:.2f} | {100 * m['ep']:.2f} | {100 * m['ttc']:.2f} | {100 * m['comfort']:.2f} | "
                  f"{100 * m['fail_nc']:.2f} | {100 * m['fail_dac']:.2f} | {100 * m['fail_ttc']:.2f} | {lead} | "
                  f"{fmt_ci(m.get('d_lead_fail'))} | {100 * m.get('frac_changed', 0):.1f} |")
    for m in out_rows:
        if "sel_share" in m:
            Lm.append("")
            Lm.append(f"{m['mode']} 선택 비율: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in m["sel_share"].items()))
    if pending:
        Lm += ["", "라벨 대기 (공식 채점 필요): " + "; ".join(pending)]
    if ref:
        Lm += ["", "참고: 이전 CK teacher navtest PDMS " + json.dumps(ref, ensure_ascii=False)]
    (td / "table.md").write_text("\n".join(Lm) + "\n")
    print((td / "table.md").read_text(), flush=True)
    print(f"[eval] -> {td} ({time.time() - t0:.1f} s)", flush=True)
    return metrics


# ----------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="CK2 teacher navtest reference A (v2 r34 top-16)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--stage", default="infer,select", help=f"comma list of {STAGES}")
    ap.add_argument("--teacher", default=None, help="ck2T | ck2M | run dir (stages infer / select / eval)")
    ap.add_argument("--which", default="last", help="checkpoint (user: last epoch)")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--limit", type=int, default=0, help="first N packed rows -> <out-root>/smoke<N> (sanity)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--var-chunk", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--labels-var", default=None)
    ap.add_argument("--labels-stack", default=None)
    a = ap.parse_args(argv)
    if a.workers > 16:
        raise SystemExit("--workers <= 16")
    stages = [s.strip() for s in a.stage.split(",") if s.strip()]
    for s in stages:
        if s not in STAGES:
            raise SystemExit(f"unknown stage {s}")
        if s != "variants" and not a.teacher:
            raise SystemExit(f"--stage {s} needs --teacher")
    for s in STAGES:
        if s in stages:
            t = time.time()
            print(f"[{kst()}] stage {s} start", flush=True)
            {"variants": stage_variants, "infer": stage_infer, "select": stage_select, "eval": stage_eval}[s](a)
            print(f"[{kst()}] stage {s} done {time.time() - t:.1f} s", flush=True)


if __name__ == "__main__":
    main()
