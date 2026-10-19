#!/usr/bin/env python
"""Fit the fixed Laplace scales of the analytic CA forecaster (SPEC 3.6, review fix D2; owner B1).  CPU only.

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace/yongjae/SSR-v3 \
    nice -n 10 /venv/ssr/bin/python -B tools/ck/v3/fit_ca_scales.py \
      [--pack D3/objpack_k96/bevfusion/navtrain_train] [--out D3/ca_scales.json] [--aug-epochs 0-9] [--seed 0] \
      [--val-pack D3/objpack_k96/bevfusion/navtrain_val]

b[g][k] = max(B_MIN = 0.05, mean over matched agent t0 slots of group g (detection class: 0 vehicle, 1 VRU) with
fc_state == OBS at step k of (|r_x| + |r_y|) / 2), r = CA(inputs) - GT (x, y) at t = 0.5 k
(objects.ca_forecast_np = a3 Worker.history + world('CA'), statics held).  M = 1, pi = 1.  navtrain_train only (refuses
another split unless --allow-split, which is for tests).

Review fix D2 (CA control fairness; protocol finding 1, correctness finding 7): the scales the model uses (key "b")
are fitted on the SAME augmented input distribution arm (b) is trained on: every train row x every epoch in
--aug-epochs (default 0-9 = the 10 epochs of the Step-1 recipe) through obj_pack.augment_objects(token, epoch, seed
= --seed (the recipe seed 0), TemporalCfg augmentation p_frame_drop / p_obj_drop / v_noise), i.e. exactly the
(token, epoch) realisations the trainer draws.  A slot counts when it survives the augmentation at t0 (the forecast
loss / metric rule).  Arms (a) and (b) therefore both use forecasters fitted on the augmented train stream and both
see the same clean inputs at validation.  The clean-input fit is kept as "b_clean" (diagnostic; the pre-fix rule),
and --val-pack adds the CA error of the clean val inputs (diagnostic only, never used for fitting).
Output: {version, source, split, k_obj, b, b_clean, n, n_clean, aug {cfg, epochs, seed}, pack_sha256 (= sha256 of the
pack's build.json), pack_obj_sha256, builder_sha256, objects_sha256, obj_pack_sha256, diagnostics, KST times}.
Never overwrites an existing file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from multiprocessing import get_context
from pathlib import Path

import numpy as np

V3 = Path(__file__).resolve().parents[3]
if str(V3) not in sys.path:
    sys.path.insert(0, str(V3))

from navsim.agents.para_ssr.ck.temporal import config as TC  # noqa: E402
from navsim.agents.para_ssr.ck.temporal import objects as O  # noqa: E402
from tools.ck.data import obj_pack as OP  # noqa: E402

KST = timezone(timedelta(hours=9))
VERSION = "v3_ca_scales_2"          # 2: augmented-stream fit (review fix D2), source / k_obj recorded
GROUPS = ("vehicle", "vru")


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def parse_epochs(s: str):
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


_P = {}


def _init(pack_dir):
    import pandas as pd
    P = Path(pack_dir)
    _P.clear()
    _P.update({k: np.load(P / f"{k}.npy", mmap_mode="r") for k in
               ("obj", "obj_mask", "frame_valid", "obj_ok", "fc_xyh", "fc_state")})
    _P["tokens"] = pd.read_parquet(P / "tokens.parquet").token.astype(str).to_numpy()


def _acc():
    return dict(S=np.zeros((2, TC.FC_STEPS)), N=np.zeros((2, TC.FC_STEPS), np.int64), E=np.zeros((2, TC.FC_STEPS)),
                Sh=np.zeros((2, 2, TC.FC_STEPS)), Nh=np.zeros((2, 2, TC.FC_STEPS), np.int64), last=[[], []])


def _add(A, obj, mask, fv, st, gt):
    sel = mask[0] & bool(fv[0]) & (st == TC.FC_OBS).any(1) & O.row_agent(obj[0])
    if not sel.any():
        return
    ca = O.ca_forecast_np(obj, mask, fv)
    grp = np.asarray(TC.GROUP_OF_CLASS)[O.row_classes(obj[0])]
    for i in np.flatnonzero(sel):
        g = int(grp[i])
        if g > 1:
            continue
        ok = st[i] == TC.FC_OBS
        d = ca["xy"][i] - gt[i, :, :2]                              # [10, 2]
        a = 0.5 * (np.abs(d[:, 0]) + np.abs(d[:, 1]))
        A["S"][g] += np.where(ok, a, 0.0)
        A["N"][g] += ok
        A["E"][g] += np.where(ok, np.hypot(d[:, 0], d[:, 1]), 0.0)
        h = int(ca["has"][i])
        A["Sh"][h, g] += np.where(ok, a, 0.0)
        A["Nh"][h, g] += ok
        if ok[-1]:
            A["last"][g].append(float(a[-1]))


def residuals(job):
    """job = (rows, epochs, seed, cfg dict) -> accumulators {'clean': .., 'aug': ..} (aug None if no epochs)."""
    rows, epochs, seed, cfgd = job
    cfg = TC.TemporalCfg.from_dict(cfgd) if cfgd else None
    C, A = _acc(), (_acc() if epochs else None)
    for r in rows:
        if not _P["obj_ok"][r]:
            continue
        it = dict(obj_in=np.array(_P["obj"][r], np.float32), obj_mask=np.array(_P["obj_mask"][r], bool),
                  frame_valid=np.array(_P["frame_valid"][r], bool))
        st = np.asarray(_P["fc_state"][r])
        gt = np.asarray(_P["fc_xyh"][r], np.float64)
        _add(C, np.asarray(it["obj_in"], np.float64), it["obj_mask"], it["frame_valid"], st, gt)
        for e in epochs:
            au = OP.augment_objects(it, str(_P["tokens"][r]), int(e), int(seed), cfg)
            _add(A, np.asarray(au["obj_in"], np.float64), au["obj_mask"], au["frame_valid"], st, gt)
    return {"clean": C, "aug": A}


def _sum(R, key):
    parts = [r[key] for r in R if r[key] is not None]
    if not parts:
        return None
    out = {k: sum(p[k] for p in parts) for k in ("S", "N", "E", "Sh", "Nh")}
    out["last"] = [np.concatenate([np.asarray(p["last"][g]) for p in parts]) for g in range(2)]
    return out


def _scales(A):
    return np.maximum(TC.B_MIN, A["S"] / np.maximum(A["N"], 1))


def _diag(A):
    return dict(
        mean_abs_half_l1={g: [round(float(x), 4) for x in (A["S"][i] / np.maximum(A["N"][i], 1))] for i, g in enumerate(GROUPS)},
        mean_euclid_err={g: [round(float(x), 4) for x in (A["E"][i] / np.maximum(A["N"][i], 1))] for i, g in enumerate(GROUPS)},
        b_with_history={g: [round(float(x), 4) for x in (A["Sh"][1, i] / np.maximum(A["Nh"][1, i], 1))] for i, g in enumerate(GROUPS)},
        b_without_history={g: [round(float(x), 4) for x in (A["Sh"][0, i] / np.maximum(A["Nh"][0, i], 1))] for i, g in enumerate(GROUPS)},
        n_with_history={g: [int(x) for x in A["Nh"][1, i]] for i, g in enumerate(GROUPS)},
        median_abs_at_5s={g: (round(float(np.median(A["last"][i])), 4) if len(A["last"][i]) else None) for i, g in enumerate(GROUPS)})


def run_pool(P: Path, n: int, epochs, seed, cfgd, workers: int, chunk: int = 512):
    chunks = [(list(range(s, min(s + chunk, n))), epochs, seed, cfgd) for s in range(0, n, chunk)]
    with get_context("fork").Pool(min(workers, len(chunks)), initializer=_init, initargs=(str(P),)) as pool:
        return pool.map(residuals, chunks, chunksize=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", default=str(TC.PACK_ROOT / "bevfusion" / "navtrain_train"))
    ap.add_argument("--out", default=str(TC.CA_SCALES))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--aug-epochs", default="0-9", help="epochs of the augmented train stream ('' = clean fit only, "
                                                         "the pre-fix rule; then 'b' = 'b_clean')")
    ap.add_argument("--seed", type=int, default=0, help="trainer --seed of the Step-1 recipe (augmentation stream)")
    ap.add_argument("--frame-drop", type=float, default=TC.TemporalCfg().p_frame_drop)
    ap.add_argument("--obj-drop", type=float, default=TC.TemporalCfg().p_obj_drop)
    ap.add_argument("--vnoise", type=float, default=TC.TemporalCfg().v_noise)
    ap.add_argument("--val-pack", default="", help="diagnostic: CA error on these clean inputs (never fitted on)")
    ap.add_argument("--limit", type=int, default=0, help="tests: first N rows only")
    ap.add_argument("--allow-split", action="store_true", help="tests only: fit on a pack other than navtrain_train")
    a = ap.parse_args(argv)
    if not 1 <= a.workers <= 16:
        raise SystemExit("--workers must be in 1..16")
    t0, k0 = time.time(), datetime.now(KST).isoformat(timespec="seconds")
    P = Path(a.pack)
    build = json.loads((P / "build.json").read_text())
    if build.get("status") != "complete" or not build.get("targets"):
        raise SystemExit(f"{P}: pack incomplete or without targets")
    if build.get("split") != "navtrain_train" and not a.allow_split:
        raise SystemExit(f"{P}: split {build.get('split')!r}: CA scales are fitted on navtrain_train only")
    if int(build.get("K_OBJ", -1)) != TC.K_OBJ:
        raise SystemExit(f"{P}: pack K_OBJ {build.get('K_OBJ')} != config K_OBJ {TC.K_OBJ}")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    epochs = parse_epochs(a.aug_epochs)
    cfg = TC.TemporalCfg(on=True, forecaster="ca", n_modes=1, p_frame_drop=a.frame_drop, p_obj_drop=a.obj_drop,
                         v_noise=a.vnoise)
    cfgd = cfg.to_dict()
    n = int(build["n"]) if not a.limit else min(int(a.limit), int(build["n"]))
    R = run_pool(P, n, epochs, a.seed, cfgd, a.workers)
    Cl, Au = _sum(R, "clean"), _sum(R, "aug")
    b_clean = _scales(Cl)
    b = _scales(Au) if Au is not None else b_clean
    res = dict(
        version=VERSION, source=build.get("source", "bevfusion"), split=build["split"], k_obj=int(build["K_OBJ"]),
        fitted_on=(f"augmented train stream: {n} rows x epochs {epochs} (augment_objects seed {a.seed}), matched agent "
                   f"t0 slots surviving the augmentation, fc_state == OBS" if Au is not None else
                   "clean pack (no augmentation), matched agent t0 slots, fc_state == OBS"),
        b={g: [round(float(x), 6) for x in b[i]] for i, g in enumerate(GROUPS)},
        b_clean={g: [round(float(x), 6) for x in b_clean[i]] for i, g in enumerate(GROUPS)},
        n={g: [int(x) for x in (Au or Cl)["N"][i]] for i, g in enumerate(GROUPS)},
        n_clean={g: [int(x) for x in Cl["N"][i]] for i, g in enumerate(GROUPS)},
        aug=dict(epochs=epochs, seed=int(a.seed), p_frame_drop=cfg.p_frame_drop, p_obj_drop=cfg.p_obj_drop,
                 v_noise=cfg.v_noise, stream=OP.AUG_STREAM) if Au is not None else None,
        FC_T=list(TC.FC_T), B_MIN=TC.B_MIN, rule="b[g][k] = max(B_MIN, mean((|r_x| + |r_y|) / 2)), r = CA - GT",
        pack=str(P), pack_sha256=sha256_file(P / "build.json"),
        pack_obj_sha256=(build.get("files") or {}).get("obj.npy", {}).get("sha256"),
        builder_sha256=sha256_file(Path(__file__).resolve()), objects_sha256=sha256_file(O.__file__),
        obj_pack_sha256=sha256_file(OP.__file__), limit=int(a.limit),
        diagnostics=dict(aug=_diag(Au) if Au is not None else None, clean=_diag(Cl),
                         ratio_aug_over_clean={g: [round(float(x), 3) for x in (b[i] / np.maximum(b_clean[i], 1e-9))]
                                               for i, g in enumerate(GROUPS)}))
    if a.val_pack:
        V = Path(a.val_pack)
        vb = json.loads((V / "build.json").read_text())
        RV = run_pool(V, int(vb["n"]) if not a.limit else min(int(a.limit), int(vb["n"])), [], 0, None, a.workers)
        Vc = _sum(RV, "clean")
        res["diagnostics"]["val_clean"] = dict(split=vb["split"], pack_sha256=sha256_file(V / "build.json"), **_diag(Vc))
    res.update(kst_start=k0, kst_end=datetime.now(KST).isoformat(timespec="seconds"), wall_s=round(time.time() - t0, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=1))
    os.replace(tmp, out)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
