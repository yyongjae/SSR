#!/usr/bin/env python
"""Store the frozen teachers' outputs on the student's training candidates = KD targets (report 44 §5; contract
pipeline.kd_targets).  This is the 'export_teacher' step of the task list.

  cd /workspace/yongjae/SSR-ck2 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/yongjae/SSR-ck2 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 /venv/ssr/bin/python tools/ck/kd_targets.py --det-run train/ckT_p1 --map-run train/ckM_p1 \
    --tag phase1 [--shard i --nshard n] [--combine-only]

Per shard (disjoint rows, shared memmaps, resumable):
  1) run_infer(det) and run_infer(map) on the packed candidates -> <out>/det/, <out>/map/ (ck_infer format, no
     self-rescoring);
  2) kd_corr_traj = correct(cand, z_DET, w_MAP, v0, slope 0).traj  (map none -> w_DET);
  3) run_infer(det / map, traj=kd_corr_traj, decode=False) -> score_logit_on_kdcorr.npy in det/ and map/.
Combine (after every shard; automatic when all rows are done, or --combine-only):
  core.kd.combine_teacher(det, map) -> kd_score_prob, kd_c_lon (DET), kd_e_lat (MAP), kd_ok;
  combine_teacher on the kd_corr scores -> kd_score_prob_corr;  meta.json (teacher run dirs, ckpt sha256, rule,
  DET teacher train-vs-val gap from its val_metrics.jsonl -> also det_gap.json).
Same training data for teachers and student (no cross-fit, user decision): the gap file records how much the DET
teacher's train-log metrics exceed its held-out navtrain-val metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[2])   # repo root of this worktree
if sys.path[0] != CK:
    sys.path.insert(0, CK)
import navsim  # noqa: E402

assert navsim.__file__.startswith(CK), navsim.__file__

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, Optional  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.ck import ckutil as U  # noqa: E402
from tools.ck.infer_ck import packed_info, resolve_run, run_infer  # noqa: E402

GAP_KEYS = ("pdms_a_b1", "pdms_a_b0.5", "pdms_v2", "pdms_oracle", "auc_fail_nc", "auc_fail_dac", "auc_fail_ttc",
            "bce_nc", "bce_dac", "bce_ep", "bce_ttc", "bce_comfort", "corr_live", "corr_e_abs")


def det_gap(det_run: Path) -> Optional[Dict]:
    """Last epoch of the DET teacher's val_metrics.jsonl: train-eval subset vs navtrain_val, gap = train - val.
    'final_epoch_ok' says whether that epoch is the run's last one (config epochs - 1, a whole epoch, both splits
    present); False -> a warning is printed and the gap belongs to an earlier model than ckpt_last.pt."""
    p = Path(det_run) / "val_metrics.jsonl"
    if not p.is_file():
        return None
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if not recs:
        return None
    last_ep = max(r["epoch"] for r in recs)
    by = {r["split"]: r for r in recs if r["epoch"] == last_ep}
    tr = next((v for k, v in by.items() if k.endswith("_train_eval")), None)
    va = by.get("navtrain_val")
    cfg = U.read_json(Path(det_run) / "config.json", {}) or {}
    exp = int(cfg["epochs"]) - 1 if cfg.get("epochs") and not cfg.get("max_steps") else None
    partial = any(bool(r.get("partial_epoch")) for r in by.values())
    final_ok = exp is not None and last_ep == exp and not partial and tr is not None and va is not None
    if not final_ok:
        print(f"WARNING det_gap {det_run}: last recorded epoch {last_ep} (expected {exp}, partial {partial}, "
              f"splits {sorted(by)}) -- the gap is not that of the final DET teacher", flush=True)
    out = {"epoch": last_ep, "expected_last_epoch": exp, "final_epoch_ok": bool(final_ok), "partial_epoch": partial,
           "train_split": None if tr is None else tr["split"], "val_split": "navtrain_val",
           "train": tr, "val": va, "history": recs}
    if tr is not None and va is not None:
        out["gap_train_minus_val"] = {k: (tr[k] - va[k]) for k in GAP_KEYS
                                      if isinstance(tr.get(k), (int, float)) and isinstance(va.get(k), (int, float))}
    return out


def kd_corr_rows(out: Path, cand, status, z_det, w_src, pos: np.ndarray, device: torch.device, bs: int = 512):
    """kd_corr_traj[pos] = correct(cand[pos], z_det[pos], w_src[pos], v0, slope 0).traj (rows with finite controls)."""
    from navsim.agents.para_ssr.ck.correct import correct
    from navsim.agents.para_ssr.refiner.e2e import ego_inputs

    N, K = cand.shape[:2]
    kc = U.open_memmap(out / "kd_corr_traj.npy", (N, K, 8, 3), np.float32, fill=np.nan)
    kd_done = U.open_memmap(out / "kd_corr_done.npy", (N,), np.bool_, fill=False)
    pos = pos[~np.asarray(kd_done[pos], bool)]
    for i in range(0, len(pos), bs):
        p = pos[i:i + bs]
        z = torch.from_numpy(np.asarray(z_det[p], np.float32)).to(device)
        w = torch.from_numpy(np.asarray(w_src[p], np.float32)).to(device)
        fin = torch.isfinite(z).flatten(1).all(1) & torch.isfinite(w).flatten(1).all(1)
        if not bool(fin.any()):
            continue
        p = p[fin.cpu().numpy()]
        z, w = z[fin], w[fin]
        c = torch.from_numpy(np.asarray(cand[p], np.float32)).to(device)
        v0 = ego_inputs(torch.from_numpy(np.asarray(status[p], np.float32)).to(device))[0]
        with torch.no_grad():
            tr = correct(c, z, w, v0, 0.0)["traj"]
        kc[p] = tr.float().cpu().numpy()
        kd_done[p] = True
    kc.flush()
    kd_done.flush()


def combine(out: Path, has_map: bool, n: int, k: int, meta_extra: Dict) -> Dict:
    from navsim.agents.para_ssr.ck import constants as Cn
    from navsim.agents.para_ssr.ck.kd import combine_teacher

    def teacher(sub: str, score_file: str, done_file: str):
        d = out / sub
        done = np.load(d / f"{done_file}.npy")
        s = np.load(d / f"{score_file}.npy").astype(np.float32)
        c = np.load(d / "c_lon.npy")
        e = np.load(d / "e_lat.npy")
        ok = done[:, None] & np.isfinite(s).all(-1) & np.isfinite(c).all(-1) & np.isfinite(e).all(-1)
        return {"score_logit": s, "c_lon": c, "e_lat": e, "ok": ok}

    det = teacher("det", "score_logit", "done")
    mp = teacher("map", "score_logit", "done") if has_map else None
    comb = combine_teacher(det, mp)
    detc = teacher("det", "score_logit_on_kdcorr", "done_on_kdcorr")
    mpc = teacher("map", "score_logit_on_kdcorr", "done_on_kdcorr") if has_map else None
    combc = combine_teacher(detc, mpc)
    kd_corr_done = np.load(out / "kd_corr_done.npy")
    arr = lambda x: x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    kd_ok = arr(comb["kd_ok"]).astype(bool) & arr(combc["kd_ok"]).astype(bool) & kd_corr_done[:, None]
    res = {"kd_score_prob": arr(comb["kd_score_prob"]).astype(np.float32),
           "kd_c_lon": arr(comb["kd_c_lon"]).astype(np.float32),
           "kd_e_lat": arr(comb["kd_e_lat"]).astype(np.float32),
           "kd_score_prob_corr": arr(combc["kd_score_prob"]).astype(np.float32),
           "kd_ok": kd_ok}
    for name, a in res.items():
        tmp = out / f".{name}.tmp{os.getpid()}.npy"
        np.save(tmp, a)
        os.replace(tmp, out / f"{name}.npy")
    meta = U.read_json(out / "meta.json", {}) or {}
    meta.update(meta_extra)
    meta.update(combined=U.now(), n=n, k=k, kd_ok_frac=float(kd_ok.mean()), kd_ok_tokens=int(kd_ok.any(1).sum()),
                rule=dict(score=dict(Cn.KD_SCORE_SOURCE), ctrl=dict(Cn.KD_CTRL_SOURCE), ctrl_w=dict(Cn.KD_CTRL_W),
                          kd_corr="correct(cand, z_det, w_map if map else w_det, v0, slope=0)"),
                stats={"kd_c_lon_mean": float(np.nanmean(res["kd_c_lon"])),
                       "kd_e_lat_abs_mean": float(np.nanmean(np.abs(res["kd_e_lat"]))),
                       "kd_score_prob_mean": [float(x) for x in np.nanmean(res["kd_score_prob"], (0, 1))],
                       "kd_score_prob_corr_mean": [float(x) for x in np.nanmean(res["kd_score_prob_corr"], (0, 1))]})
    U.write_json(out / "meta.json", meta)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description="CK KD targets (teacher outputs on the training candidates)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--det-run", required=True)
    ap.add_argument("--map-run", default="none")
    ap.add_argument("--tag", default="phase1")
    ap.add_argument("--split", default="navtrain_train")
    ap.add_argument("--out", default=None, help="default CK_DATA/kd_targets/<tag>")
    ap.add_argument("--which", default="last")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N packed rows (smoke)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--combine-only", action="store_true")
    a = ap.parse_args(argv)
    if a.workers > 16:
        raise SystemExit("--workers <= 16 per process")
    out = Path(a.out) if a.out else U.ck_data() / "kd_targets" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    det_run = resolve_run(a.det_run)
    map_run = None if a.map_run in ("none", "", None) else resolve_run(a.map_run)
    pdir, tdf, pok = packed_info(a.split)
    N_all = len(tdf)
    N = min(a.limit, N_all) if a.limit else N_all
    rows = np.arange(N, dtype=np.int64) if a.limit else None
    cand = np.load(pdir / "cand.npy", mmap_mode="r")
    status = np.load(pdir / "status.npy", mmap_mode="r")
    K = int(cand.shape[1])
    meta_extra = dict(tag=a.tag, split=a.split, det_run=str(det_run), map_run=str(map_run) if map_run else None,
                      which=a.which, packed=str(pdir), packed_meta=U.read_json(pdir / "meta.json"),
                      det_ckpt_sha256=U.sha256_file(det_run / f"ckpt_{a.which}.pt", 64),
                      map_ckpt_sha256=U.sha256_file(map_run / f"ckpt_{a.which}.pt", 64) if map_run else None,
                      limit=a.limit or None)
    if not (out / "tokens.parquet").is_file():
        sub = tdf.iloc[:N].copy().reset_index(drop=True)
        sub["row"] = np.arange(N)
        tmp = out / f".tokens.tmp{os.getpid()}.parquet"
        sub.to_parquet(tmp, index=False)
        os.replace(tmp, out / "tokens.parquet")
    gap = det_gap(det_run)
    if gap is not None:
        U.write_json(out / "det_gap.json", gap)
        meta_extra["det_gap"] = gap.get("gap_train_minus_val")
        meta_extra["det_gap_epoch"] = gap.get("epoch")
        meta_extra["det_gap_final_epoch_ok"] = gap.get("final_epoch_ok")
    t0 = time.time()
    if not a.combine_only:
        common = dict(rows=rows, which=a.which, batch=a.batch, workers=a.workers, shard=a.shard, nshard=a.nshard,
                      device=a.device)
        run_infer(det_run, a.split, out=out / "det", rescore_corr=False, **common)
        if map_run:
            run_infer(map_run, a.split, out=out / "map", rescore_corr=False, **common)
        dev = torch.device("cuda:0" if a.device.startswith("cuda") else "cpu")
        # kd_corr rows of this shard (in output-row space; output row p == packed row p since rows = arange or None)
        pos = U.shard_rows(N, a.shard, a.nshard)
        z_det = np.load(out / "det" / "z_lon.npy", mmap_mode="r")
        w_src = np.load(out / ("map" if map_run else "det") / "w_lat.npy", mmap_mode="r")
        det_done = np.load(out / "det" / "done.npy", mmap_mode="r")
        ok = np.asarray(det_done[pos], bool)
        if map_run:
            ok &= np.asarray(np.load(out / "map" / "done.npy", mmap_mode="r")[pos], bool)
        cand_n = cand[:N] if a.limit else cand
        with U.FileLock(out / ".lock_kdcorr"):
            U.open_memmap(out / "kd_corr_traj.npy", (N, K, 8, 3), np.float32, fill=np.nan)
            U.open_memmap(out / "kd_corr_done.npy", (N,), np.bool_, fill=False)
        kd_corr_rows(out, cand_n, status, z_det, w_src, pos[ok], dev)
        kc_path = out / "kd_corr_traj.npy"
        # the kd_corr_traj array is indexed by packed row (rows = arange(N)); run_infer reads traj[packed_row]
        run_infer(det_run, a.split, traj=kc_path, decode=False, score_name="score_logit_on_kdcorr",
                  done_name="done_on_kdcorr", out=out / "det", **common)
        if map_run:
            run_infer(map_run, a.split, traj=kc_path, decode=False, score_name="score_logit_on_kdcorr",
                      done_name="done_on_kdcorr", out=out / "map", **common)
        print(f"[kd] shard {a.shard}/{a.nshard} steps 1-3 in {time.time() - t0:.0f}s", flush=True)
    # combine when every packed-ok row is done in all passes
    need = np.asarray(pok[:N], bool)
    dones = [np.load(out / "det" / "done.npy"), np.load(out / "det" / "done_on_kdcorr.npy")]
    if map_run:
        dones += [np.load(out / "map" / "done.npy"), np.load(out / "map" / "done_on_kdcorr.npy")]
    missing = int(sum(int((need & ~d).sum()) for d in dones))
    if missing and not a.combine_only:
        print(f"[kd] {missing} row-passes still missing (other shards running?) -> combine later "
              f"(--combine-only)", flush=True)
        return
    if missing:
        print(f"[kd] WARNING combine with {missing} missing row-passes (kd_ok False there)", flush=True)
    meta = combine(out, map_run is not None, N, K, meta_extra)
    print(f"[kd] combined -> {out} kd_ok_frac {meta['kd_ok_frac']:.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
