#!/usr/bin/env python
"""Fit the KD probability calibration of ONE CK2 teacher run (user decision 2026-10-08 ~22:35 KST; navsim
ck/kd_calib.py): Platt scaling per KD key on the teacher's score logit, p_cal = sigmoid(a_k z_k + b_k), a_k > 0.

  # after the teacher has finished (done.json), one free GPU each (GPU inference, then CPU fitting):
  cd /workspace/yongjae/SSR-ck2 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /venv/ssr/bin/python tools/ck/e2e2/fit_kd_calib.py --run ck2T10dep --which last --n-tokens 16384
  cd /workspace/yongjae/SSR-ck2 && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /venv/ssr/bin/python tools/ck/e2e2/fit_kd_calib.py --run ck2M10dep --which last --n-tokens 16384
  -> <CK_DATA>/ck2/kd_calib/<run>__<which>.json (= the ck_e2e2.yaml kd_calib.det / .map defaults) + the inference
     cache <run>__<which>.infer.npz next to it (a rerun with the same ckpt / rows / ep_target / amp only refits).
  # CPU dry run (smoke teacher, tiny): ... CUDA_VISIBLE_DEVICES= ... --run <smoke run> --device cpu --limit 48 \
  #   --out <scratch>/calib.json

Fit data (the KD-like candidates the teacher never trained on): navtrain_train tokens of the teacher's own training
  rows (tools/ck/data/ck2_dataset.CK2Dataset with the run's config.json split / n_var / var_name / limit_tokens ->
  .rows, as train_ck2.build_datasets), N_fit = --n-tokens of them log-stratified (ckutil.log_stratified_rows, seed
  --seed; the rule of train_ck2's 'navtrain_train_train_eval' set), each with its v2 r34 top-16 executed candidates and
  their official labels (tools/ck/data/ck_dataset.CKDataset(labels='cand', ep_target = the run's config.json
  ep_target): NC / DAC / TTC official (NC 0.5 = soft target), EP = ep_target.ck_targets).  Both teachers use the
  train split (no ReSMap navtrain_val features).  A token whose teacher BEV cannot be read is dropped (the student's
  KD is off for it too); a candidate counts when its label is ok and its logits / targets are finite.
Teacher forward: model.load_ck(run, which) and the ops of online2.TeacherPair2.run (trunk.scene, trunk.candidates on
  the 16 candidates, score; fp16 autocast on CUDA when --amp 1 = ck_e2e2.teacher_amp true) -> score logits [N, 16, 5].
Temporal DET teacher (refiner-v3, its config.json temporal.on; docs/refiner_v3/STEP2_E2E.md): load_ck refuses a
  PRIVILEGED run; the fit tokens additionally get the teacher's CLEAN causal object rows of the train object pack
  (CKDataset objects = {cfg, aug False, targets False}: obj_in / obj_mask / frame_valid only, never the fc_* targets)
  and the forward is online2.teacher_forward (TemporalBranch.scene + hook = CKNet.forward(..., objs=)), the function
  TeacherPair2.run uses; tokens with obj_ok False are dropped (the student's KD is off for them too).  The inference
  cache key and the JSON 'inference' / 'data' records then also carry the temporal cfg, the pack build.json sha16 and
  the object counts.
  # v3 temporal DET teacher (GPU 0 after its training run; -> <CK_DATA>/ck2/kd_calib/v3T10_a_learned__last.json):
  cd /workspace/yongjae/SSR-v3 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /venv/ssr/bin/python tools/ck/e2e2/fit_kd_calib.py --run /home/external-user/ssd/yongjae_refiner/ck/v3/train/v3T10_a_learned \
    --which last --n-tokens 16384
Fit (CPU, float64; kd_calib.fit_platt): per KD key of this teacher (DET nc / ep / ttc, MAP dac / ep; --kd-keys), the
  soft-target BCE (NLL) minimiser (a, b), a >= 1e-3.  Held-out check: log-level 2-fold split of the fit tokens (seed
  --fold-seed): fit on one half, NLL / ECE on the other, both ways (out-of-fold); then the FINAL parameters are
  refitted on all fit tokens.  Other keys: identity, 'not_fitted' (raw metrics still reported).
JSON (kd_calib.build_record): run, which, arm, ckpt + sha16, ep_target, keys, kd_keys, fitted_keys, params {k: a, b,
  status}, metrics {k: fit {before, after}, heldout {before, after, folds}, nll_const}, data (split, n tokens /
  candidates, rows / tokens sha16, seeds, pool), inference (device, amp, cache, timing), fit_options, code (sha16 of
  this file, kd_calib.py, ep_target.py; git HEAD), created (KST).  Never writes under ck2/train or ck2/eval.
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
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from typing import Any, Dict, List, Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from navsim.agents.para_ssr.ck import kd_calib as KC  # noqa: E402
from tools.ck import ckutil as U  # noqa: E402

CK_DATA = Path("/home/external-user/ssd/yongjae_refiner/ck")
K16 = 16
KST = 9 * 3600
FORBIDDEN_OUT = ("ck2/train", "ck2/eval")
INFER_VERSION = "fit_kd_calib_infer_v1"


def kst(t: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S KST", time.gmtime((time.time() if t is None else t) + KST))


def log(msg: str) -> None:
    print(f"[{kst()}] [fit_kd_calib] {msg}", flush=True)


def sha16_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------- run / rows
def resolve_run(run: str) -> Path:
    p = Path(run)
    if p.is_dir():
        return p.absolute()
    q = CK_DATA / "ck2" / "train" / run
    if q.is_dir():
        return q
    raise SystemExit(f"teacher run not found: {run}")


def run_info(run: Path, which: str, allow_unfinished: bool = False) -> Dict[str, Any]:
    """config.json checks (trainer train_ck2, arm T / M, done.json unless allow_unfinished; refiner-v3: a temporal
    run must be a non-privileged DET teacher) -> cfg, arm, ep_target, ckpt path and sha16, temporal (TemporalCfg)."""
    from navsim.agents.para_ssr.ck.ep_target import run_ep_target
    from navsim.agents.para_ssr.ck.temporal.config import TemporalCfg
    cj = run / "config.json"
    if not cj.is_file():
        raise SystemExit(f"{run}: no config.json")
    cfg = json.loads(cj.read_text())
    if cfg.get("trainer") != "train_ck2" or cfg.get("arm") not in KC.TEACHER_OF_ARM:
        raise SystemExit(f"{run}: not a CK2 teacher run (trainer {cfg.get('trainer')!r}, arm {cfg.get('arm')!r})")
    tc = TemporalCfg.from_dict(cfg.get("temporal"))
    if tc.on and tc.privileged:
        raise SystemExit(f"{run}: PRIVILEGED temporal run (forecaster {tc.forecaster!r}, source {tc.source!r}): never a "
                         f"KD teacher, no calibration")
    if tc.on and cfg.get("arm") != "T":
        raise SystemExit(f"{run}: temporal branch on arm {cfg.get('arm')!r} (only the DET teacher, arm T)")
    if not allow_unfinished and not (run / "done.json").is_file():
        raise SystemExit(f"{run}: no done.json (teacher still training): fit the calibration after it has finished")
    ck = run / (which if which.endswith(".pt") else f"ckpt_{which}.pt")
    if not ck.is_file():
        raise SystemExit(f"{run}: {ck.name} missing")
    from navsim.agents.para_ssr.ck.e2e_data2 import teacher_v5_rel_cfg
    rel = teacher_v5_rel_cfg(run)            # refiner v5 relation DET teacher: the object pack rows it reads (else None)
    if rel is not None and tc.on:
        raise SystemExit(f"{run}: temporal branch and v5 relation channel both on (not a trainable combination)")
    return {"cfg": cfg, "arm": cfg["arm"], "ep_target": run_ep_target(cfg), "ckpt": ck,
            "ckpt_sha16": U.sha256_file(ck), "split": cfg.get("split", "navtrain_train"), "temporal": tc,
            "v5_rel": rel}


def fit_rows(cfg: Dict[str, Any], n_tokens: int, seed: int = 0, limit: int = 0, root=None):
    """-> (rows [n] int64 sorted packed rows, logs [n] str, n_pool, tokens [n] str): the teacher's training rows
    (CK2Dataset with the run's split / n_var / var_name / limit_tokens, = train_ck2.build_datasets 'train'),
    n_tokens of them log-stratified (ckutil.log_stratified_rows, seed); limit > 0 keeps `limit` rows spread evenly
    over them (dry runs)."""
    import pandas as pd

    from tools.ck.data import common as CM
    from tools.ck.data.ck2_dataset import CK2Dataset
    split = cfg.get("split", "navtrain_train")
    pool_ds = CK2Dataset(split, bev="none", n_var=int(cfg.get("n_var", 32)),
                         var_name=cfg.get("var_name", "var_separate_sampler16"), gt=False,
                         limit=int(cfg.get("limit_tokens") or 0), seed=int(cfg.get("seed", 0)),
                         sampler_seed=int(cfg.get("sampler_seed", 0)), root=root,
                         ep_target=cfg.get("ep_target") or "official")
    pool = np.asarray(pool_ds.rows, np.int64)
    tdf = pd.read_parquet(CM.packed_dir(split, root) / "tokens.parquet")
    logs_all = tdf["log"].astype(str).to_numpy()
    rows = pool[U.log_stratified_rows(logs_all[pool], int(n_tokens), seed=int(seed))]
    if limit and limit < len(rows):
        rows = rows[np.unique(np.linspace(0, len(rows) - 1, int(limit)).round().astype(np.int64))]
    return rows.astype(np.int64), logs_all[rows], int(len(pool)), tdf["token"].astype(str).to_numpy()[rows]


# ----------------------------------------------------------------------------------------------- inference
def teacher_logits(net, bev, cand, status, amp: bool, objs=None, rel_objs=None):
    """score logits f32 [B, K, 5] with the ops of online2.TeacherPair2.run (scene once, candidates, score; fp16
    autocast on CUDA when amp).  objs (refiner-v3 temporal DET teacher only: {obj_in, obj_mask, frame_valid} on the
    candidates' device) -> online2.teacher_forward, the very function TeacherPair2.run calls; None -> the CK2 ops.
    rel_objs (refiner v5 relation DET teacher only: the same three object pack keys) -> teacher_forward(rel_objs=)."""
    import torch

    from navsim.agents.para_ssr.refiner.e2e import ego_inputs
    dev = cand.device
    B, K = cand.shape[:2]
    ego = ego_inputs(status.detach().float().to(dev))
    ac = torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=bool(amp) and dev.type == "cuda")
    if (objs is not None or rel_objs is not None or getattr(net, "temporal", None) is not None
            or getattr(net, "v4_readout", None) is not None):
        from navsim.agents.para_ssr.ck.online2 import teacher_forward
        with torch.no_grad(), ac:
            s, _ = teacher_forward(net, bev.to(dev).float(), cand.float(), ego, objs, rel_objs=rel_objs)
        return s.float()
    with torch.no_grad(), ac:
        feat, mem = net.trunk.scene(bev.to(dev).float(), B)
        o = net.trunk.candidates(feat, mem, cand.float(), *ego)
        s = net.score(o, B, K)
    return s.float()


def objects_spec(tc) -> Optional[Dict[str, Any]]:
    """CKDataset `objects` kwarg of a temporal teacher (None when off): CLEAN rows, no forecast targets."""
    if tc is None or not tc.on:
        return None
    return {"cfg": tc.to_dict(), "aug": False, "targets": False}


def objects_record(tc, split: str) -> Optional[Dict[str, Any]]:
    """object pack of the fit split (dir, build.json sha16, version, n) for the cache key / JSON; None when off."""
    if tc is None or not tc.on:
        return None
    d = Path(tc.pack_root) / tc.source / split
    bj = d / "build.json"
    b = json.loads(bj.read_text()) if bj.is_file() else {}
    return {"dir": str(d), "build_sha16": U.sha256_file(bj) if bj.is_file() else None, "version": b.get("version"),
            "status": b.get("status"), "n": b.get("n"), "k_obj": b.get("K_OBJ"), "kst_end": b.get("kst_end")}


def infer(run: Path, which: str, arm: str, split: str, rows: np.ndarray, ep_target: str, device: str, batch: int,
          workers: int, amp: bool, root=None, tc=None, rel=None) -> Dict[str, Any]:
    """-> logits f32 [n, 16, 5], y f32 [n, 16, 5] (CK targets, EP per ep_target), y_ok bool [n, 16], bev_ok bool [n],
    obj_ok bool [n] (temporal teacher: object pack obj_ok; True otherwise), sanity (first batch: teacher_logits vs
    CKNet.forward max |diff|), timing.  tc: the run's TemporalCfg (on -> CLEAN object rows + temporal forward); rel:
    the v5 relation teacher's pack TemporalCfg (e2e_data2.teacher_v5_rel_cfg; -> CLEAN object rows as rel_objs, report
    99 N5: the relation channel refuses to score without them)."""
    import torch
    from torch.utils.data import DataLoader

    from navsim.agents.para_ssr.ck.model import load_ck
    from tools.ck.data.ck_dataset import CKDataset, collate_ck
    if device.startswith("cuda"):
        U.gpu_guard(device)
        dev = torch.device("cuda:0")
    else:
        dev = torch.device("cpu")
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    net, _ = load_ck(run, which=which, device=str(dev))          # refuses a PRIVILEGED run
    net.eval()
    on = getattr(net, "temporal", None) is not None
    if on != bool(tc is not None and tc.on):
        raise SystemExit(f"{run}: loaded net temporal={on} but config.json temporal.on={bool(tc is not None and tc.on)}")
    v5 = getattr(net, "v5", None)
    has_rel = bool(v5 is not None and getattr(v5, "rel", False))
    if has_rel != (rel is not None):
        raise SystemExit(f"{run}: loaded net v5 relation={has_rel} but config.json v5.relation teacher={rel is not None}")
    n = len(rows)
    out = {"logits": np.full((n, K16, 5), np.nan, np.float32), "y": np.full((n, K16, 5), np.nan, np.float32),
           "y_ok": np.zeros((n, K16), bool), "bev_ok": np.zeros(n, bool), "obj_ok": np.ones(n, bool)}
    pos = {int(r): i for i, r in enumerate(rows)}
    okw = {"objects": objects_spec(tc)} if on else {}               # CK2 teacher: no objects kwarg (old dataset)
    if has_rel:
        okw = {"objects": objects_spec(rel)}
    ds = CKDataset(split, bev=arm, k=K16, labels="cand", gt=False, rows=rows, root=root, ep_target=ep_target, **okw)
    dl = DataLoader(ds, batch_size=int(batch), shuffle=False, collate_fn=collate_ck, num_workers=int(workers),
                    pin_memory=dev.type == "cuda", prefetch_factor=4 if workers > 0 else None)
    t0, done, sanity = time.time(), 0, None
    for b in dl:
        idx = np.asarray([pos[int(r)] for r in b["rows"].numpy()], np.int64)
        cand = b["cand"].to(dev).float()
        objs = rel_objs = None
        if on or has_rel:
            from navsim.agents.para_ssr.ck.temporal.config import OBJ_INPUT_KEYS
            pack = {k: b[k].to(dev) for k in OBJ_INPUT_KEYS}            # model inputs only (no fc_* targets)
            objs, rel_objs = (pack, None) if on else (None, pack)
        s = teacher_logits(net, b["bev"], cand, b["status"], amp, objs=objs, rel_objs=rel_objs)
        if sanity is None:
            with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.float16,
                                                 enabled=bool(amp) and dev.type == "cuda"):
                ref = net(b["bev"].to(dev).float(), cand, b["status"].to(dev).float(), decode=False,
                          **({"objs": objs} if on else {}), **({"rel_objs": rel_objs} if has_rel else {}))
            sanity = {"fwd_score_maxdiff_vs_cknet": float((ref["score_logit"].float() - s).abs().max())}
        out["logits"][idx] = s.cpu().numpy()
        out["y"][idx] = b["y"].numpy().astype(np.float32)
        out["y_ok"][idx] = b["y_ok"].numpy().astype(bool)
        out["bev_ok"][idx] = b["bev_ok"].numpy().astype(bool).reshape(-1)
        if on or has_rel:
            out["obj_ok"][idx] = b["obj_ok"].numpy().astype(bool).reshape(-1)
        done += len(idx)
        if done % (int(batch) * 50) < int(batch) or done == n:
            el = time.time() - t0
            log(f"infer {done}/{n} tokens {el:.0f}s ({done / max(el, 1e-6):.1f} tok/s)")
    out["sanity"] = sanity or {}
    out["sec"] = round(time.time() - t0, 1)
    out["device"] = str(dev)
    return out


def _cache_meta(info: Dict[str, Any], rows_sha16: str, amp: bool, which: str) -> Dict[str, Any]:
    m = {"version": INFER_VERSION, "run": str(info["run"]), "which": which, "ckpt_sha16": info["ckpt_sha16"],
         "ep_target": info["ep_target"], "rows_sha16": rows_sha16, "amp": bool(amp), "labels": "cand"}
    tc = info.get("temporal")
    if tc is not None and tc.on:          # temporal teacher only: CK2 cache keys (and their caches) stay unchanged
        m["temporal"] = tc.to_dict()
        m["objects"] = objects_record(tc, info["split"])
    rel = info.get("v5_rel")
    if rel is not None:                   # v5 relation teacher only
        m["v5_rel_objects"] = objects_record(rel, info["split"])
    return m


def load_cache(path: Path, want: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    if {k: meta.get(k) for k in want} != want:
        diff = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
        log(f"inference cache {path} does not match {diff}: recomputing")
        return None
    out = {k: z[k] for k in ("logits", "y", "y_ok", "bev_ok")}
    out["obj_ok"] = z["obj_ok"] if "obj_ok" in z.files else np.ones(len(out["bev_ok"]), bool)
    out.update(sanity=meta.get("sanity", {}), sec=meta.get("sec"), device=meta.get("device"), cached=True)
    return out


def save_cache(path: Path, R: Dict[str, Any], meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp{os.getpid()}.npz")
    m = dict(meta, sanity=R.get("sanity", {}), sec=R.get("sec"), device=R.get("device"), saved=kst())
    extra = ({"obj_ok": R["obj_ok"]} if ("temporal" in meta or "v5_rel_objects" in meta)
             else {})                                                   # object-reading teachers only (CK2 npz unchanged)
    np.savez(tmp, logits=R["logits"], y=R["y"], y_ok=R["y_ok"], bev_ok=R["bev_ok"], meta=np.asarray(json.dumps(m)),
             **extra)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------- fit
def fit_all(logits: np.ndarray, y: np.ndarray, ok: np.ndarray, fold_tok: np.ndarray, arm: str,
            kd_keys: Sequence[str]) -> Dict[str, Any]:
    """logits / y [T, K, 5], ok [T, K] (candidate used), fold_tok [T] in {0, 1} -> {'params': {k: {a, b}} (fitted keys),
    'metrics': {k: ...}} for every CK key (fitted: final fit on all + out-of-fold held-out; else raw only)."""
    from navsim.agents.para_ssr.ck.constants import CK_KEYS
    fk = KC.fit_keys(arm, kd_keys)
    fold = np.broadcast_to(np.asarray(fold_tok, np.int64)[:, None], ok.shape)
    params, metrics = {}, {}
    for j, k in enumerate(CK_KEYS):
        m = ok & np.isfinite(logits[..., j]) & np.isfinite(y[..., j])
        z, t, f = logits[..., j][m].astype(np.float64), y[..., j][m].astype(np.float64), fold[m]
        rec: Dict[str, Any] = {"n": int(len(z)), "nll_const": KC.nll_const(t)}
        if k in fk:
            r = KC.fit_platt(z, t)
            params[k] = {"a": r["a"], "b": r["b"]}
            rec.update(status="fitted", fit={"before": KC.summarize(z, t), "after": KC.summarize(z, t, r["a"], r["b"]),
                                             "iters": r["iters"], "converged": r["converged"],
                                             "at_bound": r["at_bound"]},
                       heldout=KC.heldout_2fold(z, t, f))
        else:
            rec.update(status="not_fitted", fit={"before": KC.summarize(z, t)})
        metrics[k] = rec
    return {"params": params, "metrics": metrics, "fitted_keys": fk}


def code_record() -> Dict[str, Any]:
    here = Path(__file__).resolve()
    nav = Path(CK) / "navsim/agents/para_ssr/ck"
    return {"fit_kd_calib.py": U.sha256_file(here), "kd_calib.py": U.sha256_file(nav / "kd_calib.py"),
            "ep_target.py": U.sha256_file(nav / "ep_target.py"), "model.py": U.sha256_file(nav / "model.py"),
            "git_head": U.git_head(), "repo": CK}


def print_table(metrics: Dict[str, Any]) -> None:
    log("key      status      n        NLL raw -> cal (held-out raw -> oof)     ECE raw -> oof     mean p raw / cal "
        "vs label")
    for k, r in metrics.items():
        fb = r["fit"]["before"]
        if r["status"] == "fitted":
            fa, h = r["fit"]["after"], r.get("heldout", {})
            hb, ha = h.get("before", {}), h.get("after", {})
            log(f"{k:8s} fitted  {r['n']:8d}  {fb['nll']:.4f} -> {fa['nll']:.4f} ({hb.get('nll', float('nan')):.4f} -> "
                f"{ha.get('nll', float('nan')):.4f})   {hb.get('ece', float('nan')):.4f} -> "
                f"{ha.get('ece', float('nan')):.4f}   {fb['mean_pred']:.4f} / {fa['mean_pred']:.4f} vs "
                f"{fb['mean_label']:.4f}   (const NLL {r['nll_const']:.4f})")
        elif r["n"]:
            log(f"{k:8s} identity {r['n']:7d}  {fb['nll']:.4f} (not fitted)   ece {fb['ece']:.4f}   mean p "
                f"{fb['mean_pred']:.4f} vs {fb['mean_label']:.4f}")


# ----------------------------------------------------------------------------------------------- main
def get_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="teacher run name under CK_DATA/ck2/train or a run dir")
    ap.add_argument("--which", default="last")
    ap.add_argument("--n-tokens", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0, help="log-stratified row selection seed")
    ap.add_argument("--fold-seed", type=int, default=0, help="log-level 2-fold split seed (held-out check)")
    ap.add_argument("--out", default="", help="default CK_DATA/ck2/kd_calib/<run>__<which>.json")
    ap.add_argument("--kd-keys", default=",".join(KC.KD_KEYS_DEFAULT), help="= ck_e2e2.kd_score_keys")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--amp", type=int, default=1, help="fp16 autocast on CUDA (= ck_e2e2.teacher_amp true)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="dry runs: keep this many fit rows (spread evenly)")
    ap.add_argument("--force-infer", action="store_true")
    ap.add_argument("--allow-unfinished", action="store_true", help="no done.json required (tests only)")
    ap.add_argument("--data-root", default=None, help="CK data root (default CK_DATA; tests)")
    return ap


def main(argv=None) -> Dict[str, Any]:
    a = get_parser().parse_args(argv)
    t0 = time.time()
    run = resolve_run(a.run)
    info = run_info(run, a.which, a.allow_unfinished)
    info["run"] = run
    arm, ep = info["arm"], info["ep_target"]
    kd_keys = [s.strip() for s in a.kd_keys.split(",") if s.strip()]
    out = Path(a.out) if a.out else KC.calib_path(run, a.which)
    out = out.absolute()
    if any(f"/{p}/" in str(out) + "/" for p in FORBIDDEN_OUT):
        raise SystemExit(f"--out {out}: never write under {FORBIDDEN_OUT}")
    if "=" in str(out):
        raise SystemExit(f"--out {out}: '=' not allowed (Hydra override value)")
    tc = info["temporal"]
    log(f"run {run} arm {arm} ({KC.TEACHER_OF_ARM[arm]}) which {a.which} ckpt sha16 {info['ckpt_sha16']} ep_target {ep}"
        f" fitted keys {KC.fit_keys(arm, kd_keys)} -> {out}"
        + (f"; temporal teacher (forecaster {tc.forecaster}, objects {objects_record(tc, info['split'])})"
           if tc.on else ""))
    rows, logs, n_pool, toks = fit_rows(info["cfg"], a.n_tokens, a.seed, a.limit, a.data_root)
    rows_sha16 = sha16_bytes(rows.astype(np.int64).tobytes())
    log(f"fit rows: {len(rows)} tokens from {len(np.unique(logs))} logs (pool {n_pool} training rows, rows sha16 "
        f"{rows_sha16})")
    cache = out.with_name(out.stem + ".infer.npz")
    want = _cache_meta(info, rows_sha16, bool(a.amp), a.which)
    R = None if a.force_infer else load_cache(cache, want)
    if R is None:
        R = infer(run, a.which, arm, info["split"], rows, ep, a.device, a.batch, a.workers, bool(a.amp), a.data_root,
                  tc=tc, rel=info.get("v5_rel"))
        save_cache(cache, R, want)
        R["cached"] = False
        log(f"inference {R['sec']} s on {R['device']} -> {cache}; sanity {R['sanity']}")
    else:
        log(f"inference cache reused: {cache}")
    fold_tok = KC.log_folds(logs, a.fold_seed)
    ok = R["y_ok"] & R["bev_ok"][:, None]
    rel = info.get("v5_rel")
    if tc.on or rel is not None:  # object-reading DET teacher: tokens without object rows have KD off in the student too
        ok = ok & R["obj_ok"][:, None]
    F = fit_all(R["logits"], R["y"], ok, fold_tok, arm, kd_keys)
    print_table(F["metrics"])
    data = {"split": info["split"], "labels": "cand (v2 r34 top-16 executed, official labels)",
            "pool": "teacher training rows (CK2Dataset of config.json)", "n_pool": n_pool,
            "n_tokens_requested": int(a.n_tokens), "limit": int(a.limit), "n_tokens": int(len(rows)),
            "n_tokens_bev_ok": int(R["bev_ok"].sum()), "n_logs": int(len(np.unique(logs))),
            "n_candidates_ok": int(ok.sum()), "rows_sha16": rows_sha16,
            "tokens_sha16": sha16_bytes("\n".join(toks).encode()), "seed": int(a.seed), "fold_seed": int(a.fold_seed),
            "fold_tokens": [int((fold_tok == 0).sum()), int((fold_tok == 1).sum())],
            "selection": "ckutil.log_stratified_rows(log of pool rows, n_tokens, seed)" + (
                f", then {a.limit} evenly spread (dry run)" if a.limit else "")}
    inference = {"device": R.get("device"), "amp": bool(a.amp), "batch": int(a.batch), "workers": int(a.workers),
                 "forward": "online2.TeacherPair2.run ops (trunk.scene, trunk.candidates, score)",
                 "cache": str(cache), "cached": bool(R.get("cached")), "sec": R.get("sec"),
                 "sanity": R.get("sanity", {})}
    if rel is not None:          # refiner v5 relation DET teacher
        data["n_tokens_obj_ok"] = int(R["obj_ok"].sum())
        data["objects"] = objects_record(rel, info["split"])
        inference["forward"] = ("online2.teacher_forward(rel_objs = the CLEAN object pack rows) = TeacherPair2.run "
                                "(v5 relation channel)")
    if tc.on:                    # refiner-v3 temporal DET teacher (CK2 records unchanged)
        data["n_tokens_obj_ok"] = int(R["obj_ok"].sum())
        data["objects"] = objects_record(tc, info["split"])
        inference["forward"] = ("online2.teacher_forward (trunk.scene, TemporalBranch.scene + hook on the CLEAN causal "
                                "object rows, trunk.candidates, score) = TeacherPair2.run")
        inference["temporal"] = tc.to_dict()
    rec = KC.build_record(run=str(run), which=a.which, arm=arm, ckpt=str(info["ckpt"]), ckpt_sha16=info["ckpt_sha16"],
                          ep_target=ep, params=F["params"], kd_keys=kd_keys, metrics=F["metrics"], data=data,
                          inference=inference,
                          fit_options={"a_min": KC.A_MIN, "ridge": KC.RIDGE, "ece_bins": KC.N_BINS,
                                       "objective": "mean soft-target BCE (logit form), float64 Newton",
                                       "heldout": "log-level 2-fold, out-of-fold predictions; final = refit on all"},
                          code=code_record(), created=kst(), wall_sec=round(time.time() - t0, 1),
                          argv=list(sys.argv[1:] if argv is None else argv))
    U.write_json(out, rec)
    log(f"wrote {out} (sha16 {KC.file_sha16(out)}); params " + ", ".join(
        f"{k}: a {v['a']:.4f} b {v['b']:+.4f}" for k, v in F["params"].items()))
    return rec


if __name__ == "__main__":
    main()
