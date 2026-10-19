#!/usr/bin/env python
"""Per-variant-family KD calibration of ONE teacher run (refiner v4 I2, report 72: the identity-column Platt of
fit_kd_calib.py compresses the teachers' sibling gains / costs; navsim ck/kd_calib.py params_by_family).

  p_cal = sigmoid(a_{f,k} z_k + b_{f,k}),  f = the candidate's variant family (column c of the 96 pool -> c % 6:
  id, a-1.0, a-0.5, a+0.5, l-0.5, l+0.5), k = KD key of this teacher (DET nc / ep / ttc, MAP dac / ep).

  # GPU inference (one free GPU of 0-3; never 6,7), then CPU fitting:
  cd /workspace/yongjae/SSR-v4 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /venv/ssr/bin/python tools/ck/v4/fit_kd_calib_family.py --run ck2T10 --which last \
      --gen /home/external-user/ssd/yongjae_refiner/ck/ck2/e2e/ck2e2e_main/ck_e2e2/lab/ep028 --n-tokens 16384
  -> <CK_DATA>/v4/kd_calib/<run>__<which>__fam.json (+ .infer.npz); point ck_e2e2.kd_calib.det / .map at it.

Fit data: the candidates the teacher never trained on, per family: a FINISHED e2e student generation
  (<student run>/ck_e2e2/lab/epEEE: traj [N, 96, 8, 3] = the student's top-16 x 6 rule variants, official labels
  [N, 96, 9], cand_ok, valid, row_state; rows = packed navtrain_train rows), rows with row_state 1, --n-tokens of them
  log-stratified (ckutil.log_stratified_rows, --seed).  Labels -> CK targets with the teacher run's ep_target; a
  candidate counts when cand_ok & valid & finite labels / logits and the teacher BEV of its token is readable.
Teacher forward: online2.teacher_forward (= TeacherPair2.run ops) on the 96 columns, each with its parent = the
  identity column of its top-16 rank (the refiner v4 hazard readout reads the parent's corridor; CK2 teachers ignore
  it), fp16 autocast on CUDA with --amp 1.  Temporal (refiner-v3) teachers are refused (no object rows here).
Fit: fit_kd_calib.fit_all per family (kd_calib.fit_platt + log-level 2-fold held-out), record = kd_calib.build_record
  with params = the identity family's parameters and params_by_family (validated: the identity row == params).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[3])
if sys.path[0] != REPO:
    sys.path.insert(0, REPO)
import navsim  # noqa: E402

assert navsim.__file__.startswith(REPO), navsim.__file__

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from typing import Any, Dict  # noqa: E402

import numpy as np  # noqa: E402

from navsim.agents.para_ssr.ck import kd_calib as KC  # noqa: E402
from navsim.agents.para_ssr.ck.constants import CK_DATA, CK_KEYS  # noqa: E402
from tools.ck import ckutil as U  # noqa: E402
from tools.ck.e2e2 import fit_kd_calib as F  # noqa: E402

K16, NV, G_K = 16, 6, 96
OUT_ROOT = Path(CK_DATA) / "v4" / "kd_calib"


def gen_rows(gen: Path, n_tokens: int, seed: int, limit: int, root=None):
    """-> (rows [n] sorted packed rows of the generation with row_state 1, log-stratified; logs [n]; n_pool)."""
    import pandas as pd

    from tools.ck.data import common as CM
    meta = json.loads((gen / "meta.json").read_text())
    if int(meta.get("k", 0)) != G_K:
        raise SystemExit(f"{gen}: generation k {meta.get('k')} != {G_K}")
    st = np.load(gen / "row_state.npy", mmap_mode="r")
    pool = np.flatnonzero(np.asarray(st) == 1).astype(np.int64)
    if len(pool) == 0:
        raise SystemExit(f"{gen}: no finished rows (row_state 1)")
    tdf = pd.read_parquet(CM.packed_dir("navtrain_train", root) / "tokens.parquet")
    logs_all = tdf["log"].astype(str).to_numpy()
    rows = np.sort(pool[U.log_stratified_rows(logs_all[pool], int(n_tokens), seed=int(seed))])
    if limit and limit < len(rows):
        rows = rows[np.unique(np.linspace(0, len(rows) - 1, int(limit)).round().astype(np.int64))]
    return rows, logs_all[rows], int(len(pool)), meta


def infer96(run: Path, which: str, arm: str, rows: np.ndarray, gen: Path, ep_target: str, device: str, batch: int,
            workers: int, amp: bool, root=None) -> Dict[str, Any]:
    """-> logits f32 [n, 96, 5], y f32 [n, 96, 5] (CK targets), ok bool [n, 96], bev_ok bool [n], sec, device."""
    import torch
    from torch.utils.data import DataLoader

    from navsim.agents.para_ssr.ck.ep_target import ck_targets
    from navsim.agents.para_ssr.ck.model import load_ck
    from navsim.agents.para_ssr.ck.online2 import teacher_forward
    from navsim.agents.para_ssr.refiner.e2e import ego_inputs
    from tools.ck.data.ck_dataset import CKDataset, collate_ck
    if device.startswith("cuda"):
        U.gpu_guard(device)
        dev = torch.device("cuda:0")
    else:
        dev = torch.device("cpu")
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    net, _ = load_ck(run, which=which, device=str(dev))
    net.eval()
    if getattr(net, "temporal", None) is not None:
        raise SystemExit(f"{run}: temporal teacher (object rows) is not supported by the family fit")
    G = {k: np.load(gen / f"{k}.npy", mmap_mode="r") for k in ("traj", "labels", "cand_ok", "valid")}
    n = len(rows)
    out = {"logits": np.full((n, G_K, 5), np.nan, np.float32), "y": np.full((n, G_K, 5), np.nan, np.float32),
           "ok": np.zeros((n, G_K), bool), "bev_ok": np.zeros(n, bool)}
    pos = {int(r): i for i, r in enumerate(rows)}
    ds = CKDataset("navtrain_train", bev=arm, k=K16, labels="cand", gt=False, rows=rows, root=root,
                   ep_target=ep_target)
    dl = DataLoader(ds, batch_size=int(batch), shuffle=False, collate_fn=collate_ck, num_workers=int(workers),
                    pin_memory=dev.type == "cuda", prefetch_factor=4 if workers > 0 else None)
    t0, done = time.time(), 0
    ac = torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=bool(amp) and dev.type == "cuda")
    for b in dl:
        rr = b["rows"].numpy().astype(np.int64)
        idx = np.asarray([pos[int(r)] for r in rr], np.int64)
        traj = np.asarray(G["traj"][rr], np.float32)                                   # [B, 96, 8, 3]
        fin_t = np.isfinite(traj).all((-1, -2))
        traj = np.where(fin_t[..., None, None], traj, 0.0).astype(np.float32)
        y = ck_targets(np.asarray(G["labels"][rr], np.float32), ep_target)            # [B, 96, 5]
        ok = (np.asarray(G["cand_ok"][rr], bool) & np.asarray(G["valid"][rr], bool) & fin_t
              & np.isfinite(y).all(-1))
        cand = torch.from_numpy(traj).to(dev)
        par = cand[:, ::NV].repeat_interleave(NV, 1)
        ego = ego_inputs(b["status"].to(dev).float())
        with torch.no_grad(), ac:
            s, _ = teacher_forward(net, b["bev"].to(dev).float(), cand, ego, parents=par)
        out["logits"][idx] = s.float().cpu().numpy()
        out["y"][idx] = y.astype(np.float32)
        out["ok"][idx] = ok
        out["bev_ok"][idx] = b["bev_ok"].numpy().astype(bool).reshape(-1)
        done += len(idx)
        if done % (int(batch) * 50) < int(batch) or done == n:
            el = time.time() - t0
            F.log(f"infer {done}/{n} tokens {el:.0f}s ({done / max(el, 1e-6):.1f} tok/s)")
    out["sec"] = round(time.time() - t0, 1)
    out["device"] = str(dev)
    return out


def fit_families(logits: np.ndarray, y: np.ndarray, ok: np.ndarray, fold_tok: np.ndarray, arm: str, kd_keys
                 ) -> Dict[str, Any]:
    """-> {family: fit_kd_calib.fit_all(...) restricted to that family's columns}."""
    fam = np.arange(G_K) % NV
    res = {}
    for f, name in enumerate(KC.KD_FAMILIES):
        res[name] = F.fit_all(logits, y, ok & (fam == f)[None], fold_tok, arm, kd_keys)
    return res


def build_family_record(info: Dict[str, Any], run: Path, which: str, kd_keys, fits: Dict[str, Any],
                        **extra) -> Dict[str, Any]:
    fk = KC.fit_keys(info["arm"], kd_keys)
    pbf = {}
    for name in KC.KD_FAMILIES:
        P = fits[name]["params"]
        pbf[name] = {k: ({"a": float(P[k]["a"]), "b": float(P[k]["b"]), "status": "fitted"} if k in fk
                         else {"a": 1.0, "b": 0.0, "status": "not_fitted"}) for k in CK_KEYS}
    return KC.build_record(run=str(run), which=which, arm=info["arm"], ckpt=str(info["ckpt"]),
                           ckpt_sha16=info["ckpt_sha16"], ep_target=info["ep_target"],
                           params={k: fits["id"]["params"][k] for k in fk}, kd_keys=kd_keys,
                           params_by_family=pbf,
                           metrics_by_family={n: fits[n]["metrics"] for n in KC.KD_FAMILIES}, **extra)


def get_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="teacher run name under CK_DATA/ck2/train or a run dir")
    ap.add_argument("--which", default="last")
    ap.add_argument("--gen", required=True, help="finished student generation dir (<run>/ck_e2e2/lab/epEEE)")
    ap.add_argument("--n-tokens", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold-seed", type=int, default=0)
    ap.add_argument("--out", default="", help="default CK_DATA/v4/kd_calib/<run>__<which>__fam.json")
    ap.add_argument("--kd-keys", default=",".join(KC.KD_KEYS_DEFAULT))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="dry runs: keep this many rows")
    ap.add_argument("--allow-unfinished", action="store_true", help="no done.json required (tests only)")
    ap.add_argument("--data-root", default=None)
    return ap


def main(argv=None) -> Dict[str, Any]:
    a = get_parser().parse_args(argv)
    run = F.resolve_run(a.run)
    info = F.run_info(run, a.which, allow_unfinished=a.allow_unfinished)
    if info.get("temporal") is not None and getattr(info["temporal"], "on", False):
        raise SystemExit(f"{run}: temporal teacher is not supported")
    kd_keys = [k.strip() for k in a.kd_keys.split(",") if k.strip()]
    gen = Path(a.gen)
    rows, logs, n_pool, meta = gen_rows(gen, a.n_tokens, a.seed, a.limit, a.data_root)
    F.log(f"{run.name} [{info['arm']}] family fit on {len(rows)} rows of {gen} (pool {n_pool})")
    R = infer96(run, a.which, info["arm"], rows, gen, info["ep_target"], a.device, a.batch, a.workers, bool(a.amp),
                a.data_root)
    ok = R["ok"] & R["bev_ok"][:, None] & np.isfinite(R["logits"]).all(-1)
    fold = KC.log_folds(logs, a.fold_seed)
    fits = fit_families(R["logits"], R["y"], ok, np.asarray(fold, np.int64), info["arm"], kd_keys)
    out = Path(a.out) if a.out else OUT_ROOT / f"{run.name}__{a.which}__fam.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".infer.npz"), rows=rows, logits=R["logits"], y=R["y"], ok=ok)
    rec = build_family_record(
        info, run, a.which, kd_keys, fits,
        data={"gen": str(gen), "gen_epoch": meta.get("epoch"), "n_rows": int(len(rows)), "n_pool": n_pool,
              "n_cand_ok": int(ok.sum()), "per_family_ok": {n: int(ok[:, f::NV].sum())
                                                            for f, n in enumerate(KC.KD_FAMILIES)},
              "rows_sha16": F.sha16_bytes(rows.astype(np.int64).tobytes()), "seed": a.seed,
              "fold_seed": a.fold_seed},
        inference={"device": R["device"], "amp": bool(a.amp), "sec": R["sec"],
                   "forward": "online2.teacher_forward on the 96 columns (parent = identity of the top-16 rank)"},
        code={"file": str(Path(__file__).resolve()), "sha16": U.sha256_file(Path(__file__))},
        created=F.kst())
    out.write_text(json.dumps(rec, indent=1))
    F.log(f"wrote {out}")
    for f in kd_keys:
        if f not in KC.fit_keys(info["arm"], kd_keys):
            continue
        row = "  ".join(f"{n} a={rec['params_by_family'][n][f]['a']:.3f} b={rec['params_by_family'][n][f]['b']:+.3f}"
                        for n in KC.KD_FAMILIES)
        F.log(f"{f}: {row}")
    return rec


if __name__ == "__main__":
    main()
