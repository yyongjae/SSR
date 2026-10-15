#!/usr/bin/env python
"""Stage E launch helpers (no GPU, no training).

  snapshot : read-only copy of the frozen teacher refiners (config.json, norm*.npz, ckpt_best.pt) + sha256.json
             python tools/refiner/stageE_prep.py snapshot [--runs <run_dir> ...] [--out <dir>]
  pilot    : pilot summary from <exp>/stageE_steps.jsonl: seconds per micro-batch (compute + data wait), peak GPU
             memory, surrogate / KD magnitudes and the lambda_KD rule
             lambda_c = mean(w_ref * L_sur) / mean(L_KD) over the LAST 100 micro-batches of the E2 pilot
             python tools/refiner/stageE_prep.py pilot --steps <E2 exp>/stageE_steps.jsonl [--other <E1 jsonl>]
  epochs   : N = largest integer with launch + startup + N * margin * sec_per_mb * mb_per_epoch <= deadline
             (margin 1.15, startup 10 min by default: E0's epochs varied 2,621-5,160 s under host contention and
             checkpoints are saved only at epoch ends, so a crash costs up to one epoch)
             python tools/refiner/stageE_prep.py epochs --sec-per-mb 0.5 --mb-per-epoch 21278 \
                    --launch "2026-09-29 19:45" --deadline "2026-09-30 18:00"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

RUNS = Path("/home/external-user/ssd/yongjae_refiner/runs")
TEACHERS = [RUNS / "stageT4_T_fold0_seed0", RUNS / "stageT4_M_fold0_seed0"]
SNAP = Path("/home/external-user/ssd/yongjae_refiner/stageE/teachers")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def snapshot(a) -> None:
    out = Path(a.out)
    rec = {}
    for run in map(Path, a.runs):
        dst = out / run.name
        if dst.exists():
            raise SystemExit(f"{dst} exists (snapshots are write-once)")
        files = ["config.json", "ckpt_best.pt"] + sorted(p.name for p in run.glob("norm*.npz"))
        if not (run / "DONE").exists() and not a.allow_unfinished:
            raise SystemExit(f"{run} has no DONE marker (still training?); --allow-unfinished for a smoke snapshot")
        dst.mkdir(parents=True)
        for f in files:
            shutil.copy2(run / f, dst / f)
            (dst / f).chmod(0o444)
        rec[run.name] = {"src": str(run), "files": {f: sha256(dst / f) for f in files},
                         "done": (run / "DONE").read_text().strip() if (run / "DONE").exists() else None,
                         "summary": json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else None}
    out.mkdir(parents=True, exist_ok=True)
    p = out / "sha256.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(rec)
    p.write_text(json.dumps(old, indent=1))
    print(json.dumps(rec, indent=1))


def _load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def summarise(recs, last: int = 100, skip: int = 20) -> dict:
    body = recs[skip:] if len(recs) > skip + 10 else recs
    step = np.array([r["sec_step"] for r in body if r.get("sec_step") is not None])
    wait = np.array([r["sec_wait"] for r in body if r.get("sec_wait") is not None])
    tail = recs[-last:]
    m = lambda k: float(np.mean([r[k] for r in tail if k in r])) if any(k in r for r in tail) else None
    out = dict(n=len(recs), sec_step_median=float(np.median(step)) if len(step) else None,
               sec_wait_median=float(np.median(wait)) if len(wait) else None,
               sec_per_mb_mean=float(np.mean(step) + (np.mean(wait) if len(wait) else 0.0)) if len(step) else None,
               mem_gb_peak=max([r.get("mem_gb", 0.0) for r in recs] or [0.0]),
               tail_n=len(tail), L_sur_weighted=m("ref/L_sur_weighted"), L_KD=m("kd/loss"),
               kd_l1_0=m("kd/l1_0"), kd_l1_1=m("kd/l1_1"), live=m("ref/live"), teacher_live_0=m("kd/teacher_live_0"),
               teacher_live_1=m("kd/teacher_live_1"), ref_ms=m("time/ref_ms"), perturb_ms=m("time/ref_perturb_ms"),
               gt_ok_frac=(m("ref/n_gt_ok") / (m("ref/n_gt_ok") + m("ref/n_gt_missing"))) if m("ref/n_gt_ok") is not None else None)
    if out["L_sur_weighted"] is not None and out["L_KD"]:
        out["lambda_c"] = out["L_sur_weighted"] / out["L_KD"]
    # post-pilot options (present only when logged): EMA weight, weighted KD, value / BEV-gradient shares, human drafts
    for k in ("kd/lambda", "kd/weighted", "kd/w_ema", "kd/ema_sur", "kd/ema_kd", "loss_e0", "ref/vshare_e0",
              "ref/vshare_sur", "ref/vshare_kd", "ref/gshare_e0", "ref/gshare_sur", "ref/gshare_kd", "gnorm/bev_e0",
              "gnorm/bev_sur", "gnorm/bev_kd", "ref/frac_human", "ref/n_human_invalid"):
        v = [r[k] for r in tail if k in r]
        if v:
            out[k] = float(np.mean(v))
    return out


def pilot(a) -> None:
    res = {"E2" if a.other else "run": summarise(_load(a.steps), a.last, a.skip)}
    if a.other:
        res["E1"] = summarise(_load(a.other), a.last, a.skip)
    print(json.dumps(res, indent=1))


def epochs(a) -> None:
    t0 = datetime.strptime(a.launch, "%Y-%m-%d %H:%M")
    t1 = datetime.strptime(a.deadline, "%Y-%m-%d %H:%M")
    nominal = a.sec_per_mb * a.mb_per_epoch
    per_epoch = nominal * a.margin
    avail = (t1 - t0).total_seconds() - 60.0 * a.startup_min
    n = int(math.floor(avail / per_epoch))
    print(json.dumps(dict(epoch_hours=nominal / 3600.0, epoch_hours_with_margin=per_epoch / 3600.0, N=n,
                          end_nominal=str(t0 + timedelta(seconds=60.0 * a.startup_min + n * nominal)),
                          end_with_margin=str(t0 + timedelta(seconds=60.0 * a.startup_min + n * per_epoch)))))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("snapshot")
    s.add_argument("--runs", nargs="+", default=[str(p) for p in TEACHERS])
    s.add_argument("--out", default=str(SNAP))
    s.add_argument("--allow-unfinished", action="store_true")
    s = sp.add_parser("pilot")
    s.add_argument("--steps", required=True)
    s.add_argument("--other", default=None)
    s.add_argument("--last", type=int, default=100)
    s.add_argument("--skip", type=int, default=20)
    s = sp.add_parser("epochs")
    s.add_argument("--sec-per-mb", type=float, required=True)
    s.add_argument("--mb-per-epoch", type=int, default=21278)
    s.add_argument("--launch", required=True)
    s.add_argument("--deadline", default="2026-09-30 18:00")
    s.add_argument("--margin", type=float, default=1.15)
    s.add_argument("--startup-min", type=float, default=10.0)
    a = ap.parse_args(argv)
    {"snapshot": snapshot, "pilot": pilot, "epochs": epochs}[a.cmd](a)


if __name__ == "__main__":
    main()
