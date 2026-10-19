#!/usr/bin/env python
"""Fit the frozen teacher statistics of the refiner v5 student's Z adapters (ck_e2e2.v4.v5.z_adapt; navsim
ck/v5/zadapt.py; design refiner_v5_design_final.md §3.8 L191 "mu_T, sigma_T the frozen per-channel statistics of the
teacher reader on navtrain (buffers, as bev_kd norm.npz)"): ONE finished v5 teacher run -> a JSON record of the per-channel
mean / std of its per-state reads Z (the zq of score_set: DET teacher b_nc, b_ttc, H_nc, H_ttc; MAP teacher d_min).

  # after the teacher has finished (done.json); CPU (measured ~0.4-1 s / token at 96 columns, OMP 8 faster):
  cd /workspace/yongjae/SSR-v5 && CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    /venv/ssr/bin/python tools/ck/v5/fit_z_stats.py --run v5T_rel4 --which last --tokens 4096
  cd /workspace/yongjae/SSR-v5 && CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    /venv/ssr/bin/python tools/ck/v5/fit_z_stats.py --run v5M_unk3 --which last --tokens 4096
  -> <CK_DATA>/v5/z_stats/<run>__<which>.json (set ck_e2e2.v4.v5.z_stats_det / z_stats_map to these files; the
     launcher refuses the arm file's /PLACEHOLDER/ while v4.v5.z_adapt is on, and dry-run records in strict mode).
  # dry run (64 tokens, a DRY-RUN record: accepted by the student build / CPU smoke, refused by a strict launch):
  ... tools/ck/v5/fit_z_stats.py --run v5T_rel4 --tokens 64 --out <scratch>/z_stats_det.json
  # --device cuda (one free GPU of 0-3 with the user's approval; fp16 autocast = ck_e2e2.teacher_amp with --amp 1).

Run (--run NAME | DIR; a name is looked up under CK/v5/train, CK/v4/train, CK/ck2/train): a train_ck2 run of arm T or M
  whose net has the v5 per-state reads (V5Cfg on: the range readout's zq); done.json required unless
  --allow-unfinished (the record is then flagged 'unfinished' and every consumer refuses it).  The checkpoint's sha16 is
  taken before and after the load (a changing ckpt_last.pt is refused).  A privileged run is refused (model.load_ck).
Data: navtrain_train tokens of the teacher's own training rows (the fit_kd_calib.fit_rows pool = train_ck2.build_datasets
  'train'; select_rows: exactly --tokens of them, log-stratified and topped up per log, seed --seed; --limit N keeps N
  of those spread evenly), each with its v2 r34
  top-16 executed candidates and their online CK2 variants = the 96 deployment pool (cands2.online_variants with the
  CK2 e2e variants config, online2.infer_pool: candidates, finite mask, parents) -- a superset of the e2e student's KD set
  (top-16 + 32 sampled variants) on which the Z-KD targets are read.  BEV: the teacher's own (T: BEVFusion cache, M:
  ReSMap cache); a relation DET teacher also reads its CLEAN object-pack rows (fit_kd_calib.objects_spec).
Forward: online2.teacher_forward(net, bev, cand96, ego, parents=par96, rel_objs, extras) = TeacherPair2.run's call
  (extras['zq'] = score_set haz zq [B, 96, Q', C]); float32 on CPU, fp16 autocast on CUDA with --amp 1.
Statistics (float64; zadapt.ZStatsAcc / stats_mask): per channel of zadapt.TEACHER_CHANNELS[arm] over the valid finite
  columns (valid96 & finite, bev_ok, obj_ok), known states (unknown <= 0.5) and, for b_nc / b_ttc, READ states (zq above
  the -4 floor: 'not read' is a flag, not a logit) of every strided state; std floored at zadapt.STD_FLOOR; a channel
  with fewer than zadapt.N_MIN states is refused (SystemExit).  Floor share / min / max recorded.
JSON (zadapt.STATS_VERSION): run (resolved), run_name, which, arm, teacher det | map, ckpt + ckpt_sha16, v5 cfg, grid
  (z_stride, ev_tbins, ev_tbin_s, rg_read, rg_stopped_aware, corridor_version), channels {name: mean, std, std_raw, n,
  min, max, status (+ floor_share, floor_normalised)}, mask_rule, data (split, n_tokens, rows / tokens sha16, seed,
  limit, pool, variants), inference (device, amp, batch, sec, tok_s), code sha16 (this file, zadapt.py, zread.py,
  relation.py, model.py; git HEAD), created (KST), unfinished, dry_run (n_tokens < zadapt.DRY_RUN_BELOW or --limit).
  Never writes under a train / eval run dir (ck2/train, ck2/eval, v5/train, v4/train).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[3])   # repo root of this worktree
if sys.path[0] != REPO:
    sys.path.insert(0, REPO)
import navsim  # noqa: E402

assert navsim.__file__.startswith(REPO), navsim.__file__

import argparse  # noqa: E402
import hashlib  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from typing import Any, Dict, Optional  # noqa: E402

import numpy as np  # noqa: E402

from navsim.agents.para_ssr.ck.v5 import zadapt as ZA  # noqa: E402
from tools.ck import ckutil as U  # noqa: E402
from tools.ck.e2e2 import fit_kd_calib as FK  # noqa: E402

CK_DATA = Path("/home/external-user/ssd/yongjae_refiner/ck")
RUN_ROOTS = ("v5/train", "v4/train", "ck2/train")
FORBIDDEN_OUT = ("ck2/train", "ck2/eval", "v5/train", "v4/train", "v5/eval", "v4/eval")
K16 = 16
KST = 9 * 3600
USAGE = ("CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /venv/ssr/bin/python "
         "tools/ck/v5/fit_z_stats.py --run <v5 teacher run> --which last --tokens 4096 [--out <json>]")


def kst(t: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S KST", time.gmtime((time.time() if t is None else t) + KST))


def log(msg: str) -> None:
    print(f"[{kst()}] [fit_z_stats] {msg}", flush=True)


def sha16_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def resolve_run(run: str) -> Path:
    p = Path(run)
    if p.is_dir():
        return p.absolute()
    for r in RUN_ROOTS:
        q = CK_DATA / r / run
        if q.is_dir():
            return q
    raise SystemExit(f"teacher run not found: {run} (a dir, or a name under {[str(CK_DATA / r) for r in RUN_ROOTS]})")


def default_out(run: Path, which: str) -> Path:
    w = Path(which).stem if which.endswith(".pt") else which
    return CK_DATA / "v5" / "z_stats" / f"{run.name}__{w}.json"


def check_out(out: Path) -> Path:
    out = out.absolute()
    s = str(out) + "/"
    if any(f"/{p}/" in s for p in FORBIDDEN_OUT):
        raise SystemExit(f"--out {out}: never write under a train / eval run dir {FORBIDDEN_OUT}")
    if "=" in str(out):
        raise SystemExit(f"--out {out}: '=' not allowed (Hydra override value)")
    return out


def grid_of(cfg: Dict[str, Any], v5) -> Dict[str, Any]:
    v4 = cfg.get("v4") or {}
    return {"z_stride": int(v5.z_stride), "ev_tbins": v4.get("ev_tbins"), "ev_tbin_s": v4.get("ev_tbin_s"),
            "rg_read": v4.get("rg_read"), "rg_stopped_aware": v4.get("rg_stopped_aware"),
            "corridor_version": int(cfg.get("corridor_version") or 1), "exec_model": v4.get("exec_model")}


def select_rows(logs: np.ndarray, n: int, seed: int = 0, limit: int = 0) -> np.ndarray:
    """indices into the pool (logs [N] of the teacher's training rows) of exactly min(n, N) fit tokens:
    ckutil.log_stratified_rows (the same fraction of every log; fit_kd_calib's rule), topped up -- it undershoots for a
    small n, where round(n / N x len(log)) is 0 for most logs -- with one unchosen row of each log in turn (logs in a
    seeded order); limit > 0 then keeps `limit` of them spread evenly (dry runs).  Sorted."""
    logs = np.asarray(logs)
    N = len(logs)
    n = min(int(n), N) if n > 0 else N
    sel = U.log_stratified_rows(logs, n, seed=int(seed))
    if len(sel) < n:
        rng = np.random.default_rng(int(seed) + 1)
        taken = np.zeros(N, bool)
        taken[sel] = True
        uniq = np.unique(logs)
        order = uniq[rng.permutation(len(uniq))]
        per = {lg: list(rng.permutation(np.flatnonzero((logs == lg) & ~taken))) for lg in order}
        extra = []
        while len(sel) + len(extra) < n:
            moved = False
            for lg in order:
                if per[lg] and len(sel) + len(extra) < n:
                    extra.append(int(per[lg].pop()))
                    moved = True
            if not moved:
                break
        sel = np.sort(np.concatenate([sel, np.asarray(extra, np.int64)]))
    if limit and limit < len(sel):
        sel = sel[np.unique(np.linspace(0, len(sel) - 1, int(limit)).round().astype(np.int64))]
    return sel.astype(np.int64)


def load_teacher(run: Path, which: str, device: str):
    """model.load_ck with the checkpoint's sha16 before and after the load (equal, or SystemExit)."""
    from navsim.agents.para_ssr.ck.model import load_ck
    ck = ZA.teacher_ckpt(run, which)
    if not ck.is_file():
        raise SystemExit(f"{run}: {ck.name} missing")
    h0 = U.sha256_file(ck)
    try:
        net, cfg = load_ck(run, which=which, device=device)          # refuses a PRIVILEGED run
    except PermissionError as e:
        raise SystemExit(str(e)) from None
    h1 = U.sha256_file(ck)
    if h0 != h1:
        raise SystemExit(f"{ck}: the checkpoint changed during the load ({h0} -> {h1}; a running training rewrote it)")
    v5 = getattr(net, "v5", None)
    if v5 is None or not v5.on or getattr(net, "v4_readout", None) is None:
        raise SystemExit(f"{run}: not a refiner v5 net (config.json v5 off): it has no per-state reads Z")
    return net.eval(), cfg, ck, h0


def run_stats(net, arm: str, split: str, rows: np.ndarray, device, batch: int, workers: int, amp: bool,
              rel_tc=None, vcfg: Optional[Dict[str, Any]] = None, root=None) -> Dict[str, Any]:
    """the 96-pool teacher forward over rows -> {acc (ZStatsAcc), n_bev_ok, n_obj_ok, n_cols_ok, sec, Q, C}."""
    import torch
    from torch.utils.data import DataLoader

    from navsim.agents.para_ssr.ck import cands2 as C2
    from navsim.agents.para_ssr.ck.online2 import infer_pool, teacher_forward
    from navsim.agents.para_ssr.refiner.e2e import ego_inputs
    from tools.ck.data.ck_dataset import CKDataset, collate_ck
    dev = torch.device(device)
    has_rel = getattr(net, "relation", None) is not None
    if has_rel != (rel_tc is not None):
        raise SystemExit(f"loaded net relation={has_rel} but config.json v5.relation teacher={rel_tc is not None}")
    okw = {"objects": FK.objects_spec(rel_tc)} if has_rel else {}
    ds = CKDataset(split, bev=arm, k=K16, labels=None, gt=False, rows=rows, root=root, **okw)
    dl = DataLoader(ds, batch_size=int(batch), shuffle=False, collate_fn=collate_ck, num_workers=int(workers),
                    prefetch_factor=4 if workers > 0 else None)
    acc = ZA.ZStatsAcc(ZA.TEACHER_CHANNELS[arm])
    n_bev = n_obj = n_cols = done = 0
    shape = None
    t0 = time.time()
    ac = torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=bool(amp) and dev.type == "cuda")
    for b in dl:
        cand16 = b["cand"].to(dev).float()
        status = b["status"].to(dev).float()
        V = C2.online_variants(cand16, C2.v0_from_status(status), vcfg)
        cand96, fin96, par96 = infer_pool(V["traj96"], cand16)
        ego = ego_inputs(status)
        bev_ok = b["bev_ok"].to(dev).bool().reshape(-1)
        obj_ok = torch.ones_like(bev_ok)
        rel_objs = None
        if has_rel:
            from navsim.agents.para_ssr.ck.temporal.config import OBJ_INPUT_KEYS
            rel_objs = {k: b[k].to(dev) for k in OBJ_INPUT_KEYS}           # model inputs only (no fc_* targets)
            obj_ok = b["obj_ok"].to(dev).bool().reshape(-1)
        extras: Dict[str, Any] = {}
        with torch.no_grad(), ac:
            teacher_forward(net, b["bev"].to(dev).float(), cand96, ego, parents=par96, rel_objs=rel_objs,
                            extras=extras)
        if "zq" not in extras:
            raise SystemExit("teacher_forward returned no zq (not a v5 net?)")
        zq = extras["zq"].float()                                           # [B, 96, Q', C]
        col_ok = V["valid96"].to(dev) & fin96 & (bev_ok & obj_ok)[:, None]
        acc.update(zq, col_ok)
        shape = tuple(zq.shape[2:])
        n_bev += int(bev_ok.sum())
        n_obj += int(obj_ok.sum())
        n_cols += int(col_ok.sum())
        done += int(cand16.shape[0])
        if done % (int(batch) * 25) < int(batch) or done == len(rows):
            el = time.time() - t0
            log(f"{done}/{len(rows)} tokens {el:.0f} s ({done / max(el, 1e-6):.2f} tok/s)")
    return {"acc": acc, "n_bev_ok": n_bev, "n_obj_ok": n_obj, "n_cols_ok": n_cols, "sec": round(time.time() - t0, 1),
            "Q": None if shape is None else int(shape[0]), "C": None if shape is None else int(shape[1])}


def code_record() -> Dict[str, Any]:
    nav = Path(REPO) / "navsim/agents/para_ssr/ck"
    return {"fit_z_stats.py": U.sha256_file(Path(__file__).resolve()), "zadapt.py": U.sha256_file(nav / "v5/zadapt.py"),
            "zread.py": U.sha256_file(nav / "v5/zread.py"), "relation.py": U.sha256_file(nav / "v5/relation.py"),
            "model.py": U.sha256_file(nav / "model.py"), "git_head": U.git_head(), "repo": REPO}


def get_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], epilog="usage: " + USAGE)
    ap.add_argument("--run", required=True, help="teacher run dir or name (CK/v5/train, CK/v4/train, CK/ck2/train)")
    ap.add_argument("--which", default="last")
    ap.add_argument("--tokens", "--n-tokens", dest="tokens", type=int, default=4096,
                    help="navtrain fit tokens (log-stratified teacher training rows); < %d = a dry-run record"
                         % ZA.DRY_RUN_BELOW)
    ap.add_argument("--seed", type=int, default=0, help="log-stratified row selection seed")
    ap.add_argument("--limit", type=int, default=0, help="smoke: keep this many of the rows (a dry-run record)")
    ap.add_argument("--out", default="", help="default CK_DATA/v5/z_stats/<run>__<which>.json")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--amp", type=int, default=1, help="fp16 autocast on CUDA (= ck_e2e2.teacher_amp true)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--allow-unfinished", action="store_true",
                    help="no done.json required (smoke only: the record is flagged unfinished and refused by consumers)")
    ap.add_argument("--data-root", default=None, help="CK data root (default CK_DATA; tests)")
    return ap


def main(argv=None) -> Dict[str, Any]:
    import torch
    a = get_parser().parse_args(argv)
    t0 = time.time()
    run = resolve_run(a.run)
    info = FK.run_info(run, a.which, a.allow_unfinished)          # train_ck2 arm T / M, done.json, v5_rel pack cfg
    arm = info["arm"]
    if arm not in ZA.TEACHER_CHANNELS:
        raise SystemExit(f"{run}: arm {arm!r} (z statistics are fitted for the DET (T) / MAP (M) teachers)")
    if info["temporal"].on:
        raise SystemExit(f"{run}: a temporal-branch teacher has no v5 per-state reads")
    out = check_out(Path(a.out) if a.out else default_out(run, a.which))
    if a.device == "cuda":
        U.gpu_guard("cuda")
        device = "cuda:0"
    else:
        device = "cpu"
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    net, cfg, ck, ck_sha16 = load_teacher(run, a.which, device)
    log(f"run {run} arm {arm} ({ZA.TEACHER_OF_ARM[arm]}) which {a.which} ckpt sha16 {ck_sha16} channels "
        f"{ZA.TEACHER_CHANNELS[arm]} -> {out}")
    pool, pool_logs, n_pool, pool_toks = FK.fit_rows(info["cfg"], 0, a.seed, 0, a.data_root)   # every training row
    sel = select_rows(pool_logs, a.tokens, a.seed, a.limit)
    rows, logs_, toks = pool[sel], pool_logs[sel], pool_toks[sel]
    rows_sha16 = sha16_bytes(rows.astype(np.int64).tobytes())
    log(f"rows: {len(rows)} tokens from {len(np.unique(logs_))} logs (pool {n_pool} training rows, sha16 {rows_sha16})")
    from navsim.agents.para_ssr.ck.online2 import _d_variants
    vcfg = _d_variants()
    R = run_stats(net, arm, info["split"], rows, device, a.batch, a.workers, bool(a.amp), rel_tc=info.get("v5_rel"),
                  vcfg=vcfg, root=a.data_root)
    dry = bool(a.limit) or len(rows) < ZA.DRY_RUN_BELOW
    try:
        chans = R["acc"].channels(ZA.N_MIN, allow_small=False)
    except ValueError as e:
        raise SystemExit(f"{run}: {e} (more --tokens)") from None
    for n, c in chans.items():
        log(f"  {n:6s} mean {c['mean']:+.4f} std {c['std']:.4f} n {c['n']}"
            + (f" floor share {c['floor_share']:.4f} (floor at {c['floor_normalised']:+.2f} sd)"
               if c.get("floor_share") is not None else "")
            + f" range [{c['min']:+.3f}, {c['max']:+.3f}]")
    rec = {"version": ZA.STATS_VERSION, "run": str(run), "run_realpath": os.path.realpath(str(run)),
           "run_name": run.name, "which": a.which, "arm": arm, "teacher": ZA.TEACHER_OF_ARM[arm], "ckpt": str(ck),
           "ckpt_sha16": ck_sha16, "v5": net.v5.to_dict(),
           "grid": grid_of(cfg, net.v5), "zq_layout": list(ZA.Z_CHANNELS[:R["C"] or 0]), "Q": R["Q"],
           "channels": chans, "mask_rule": ZA.MASK_RULE, "std_floor": ZA.STD_FLOOR, "n_min": ZA.N_MIN,
           "data": {"split": info["split"], "n_tokens_requested": int(a.tokens), "limit": int(a.limit),
                    "n_tokens": int(len(rows)), "n_tokens_bev_ok": R["n_bev_ok"], "n_tokens_obj_ok": R["n_obj_ok"],
                    "n_columns_ok": R["n_cols_ok"], "n_logs": int(len(np.unique(logs_))), "n_pool": n_pool,
                    "rows_sha16": rows_sha16, "tokens_sha16": sha16_bytes("\n".join(toks).encode()),
                    "seed": int(a.seed), "pool": "v2 r34 top-16 + online CK2 variants (96, online2.infer_pool)",
                    "variants": vcfg, "rows": "teacher training rows (fit_kd_calib.fit_rows pool), select_rows: "
                                              "log-stratified + per-log top-up" + (f", then {a.limit} evenly spread"
                                                                                   if a.limit else ""),
                    "objects": FK.objects_record(info["v5_rel"], info["split"]) if info.get("v5_rel") else None},
           "inference": {"device": device, "amp": bool(a.amp) and device.startswith("cuda"), "batch": int(a.batch),
                         "workers": int(a.workers), "sec": R["sec"],
                         "tok_s": round(len(rows) / max(R["sec"], 1e-6), 3),
                         "forward": "online2.teacher_forward(parents = infer_pool parents, rel_objs) = TeacherPair2.run"},
           "code": code_record(), "created": kst(), "wall_sec": round(time.time() - t0, 1),
           "unfinished": not (run / "done.json").is_file(), "dry_run": dry,
           "argv": list(sys.argv[1:] if argv is None else argv)}
    U.write_json(out, rec)
    log(f"wrote {out} (sha16 {ZA.file_sha16(out)}){' DRY RUN' if dry else ''}{' UNFINISHED' if rec['unfinished'] else ''}")
    return rec


if __name__ == "__main__":
    main()
