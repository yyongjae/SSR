#!/usr/bin/env python
"""CK Phase 2 evaluation of a trained v2 + CK e2e checkpoint (contract launch.eval_e2e.py; report 45 §2-4).

  cd /workspace/yongjae/SSR-ck2 && PYTHONPATH=/workspace/yongjae/SSR-ck2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /venv/ssr/bin/python tools/ck/e2e/eval_e2e.py --run-dir <RUN> --split navtrain_val [--ckpt last] [--limit N] \
    [--gpus 0,1,2,3] [--stage extract,dump,label,infer_check,eval,summary | all] [--workers-label 48] \
    [--fix-from <val metrics.json>] [--fresh]

Stages (each resumable; --stage all runs them in this order):
  extract      Lightning ckpt 'agent.ck_student.*' -> <RUN>/eval/ck_run/{config.json, ckpt_last.pt} (ck.model.save_ck
               format; arm S, score_hidden 256, lead_aux 0, e2e true) and a strict ck.model.load_ck round trip.
               The resolved ckpt is pinned in <RUN>/eval/ckpt.json for the later stages.
  dump         agent from <RUN>/train/code/hydra/config.yaml + ckpt (run_aux_evaluation._build_agent, strict load),
               eval-mode forward (ck_* outputs of CKE2E.infer; trajectory = v2) on the split's tokens; SceneLoader /
               feature builders as tools/ck/data/dump_v2.run_shard (imported, dump_v2.py unchanged, no v2 sha assert).
               Writes the eval root <RUN>/eval/root (used as CK_DATA_ROOT):
                 packed/<split>/{tokens.parquet, cand, cand_idx, v2_final, v2_im, v2_sim, status, gt_traj, sub_traj, ok,
                                 ok_rows, meta.json}                                   (pack_v2 format)
                 infer/<RUN name>/<split>/{score_logit, z_lon, w_lat, c_lon, e_lat, corr_traj, corr_score_logit, done,
                                 tokens.parquet, meta.json, check_bev.npz}            (infer_ck format)
               asserts tau[:, 0] == trajectory (1e-4) and ck_cand == gather(anchors + offset, topk16(final)) (1e-5).
               Several GPUs: one child process per GPU (whole logs per shard, dump_v2.shard_tokens).
  label        CK_DATA_ROOT=root label_cands.py --name cand, then --name corr_<RUN name> --traj infer/.../corr_traj.npy
               (official score_token / LQR; rows = packed ok; workers <= 48).
  infer_check  the extracted ck_run (load_ck, fp32, slope 0) on 8 stored BEV grids reproduces the in-process ck_*.
  eval         CK_DATA_ROOT=root eval_ck.py: navtrain_val = beta grid / variant selection; navtest only with
               --fix-from (default root/eval/<RUN>/navtrain_val/metrics.json): chosen beta + beta 1.  beta / variants are
               never chosen on navtest.  The variant is chosen on navtrain_val too (select_variant: of a/b/c, the
               highest PDMS at its own val beta, ties -> a, b, c) and stored as 'best_variant' in the val metrics.json;
               the navtest metrics.json carries it as 'best_variant_from_val'.
  summary      <RUN>/eval/summary.{json,md}: per split v2-head (beta 0), the val-chosen variant at its val beta (the
               representative row, marked), oracle16, the other variants at the val beta and beta 1 (descriptive),
               lead-decel NC|TTC failure, Delta vs v2-head with log-cluster bootstrap CI, plus the reference v2 r34
               (Phase 1 labels of r34 cand 0 on the same tokens; 2 GPUs, other seed: confounded, descriptive only).
navtest 'v2_mismatch' from eval_ck compares with the r34 CSV and does not apply to an e2e checkpoint (noted).
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
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from tools.ck import ckutil as U  # noqa: E402

PY = "/venv/ssr/bin/python"
CK_DATA = Path("/home/external-user/ssd/yongjae_refiner/ck")
K = 16
STUDENT_PREFIX = "agent.ck_student."
STAGES = ("extract", "dump", "label", "infer_check", "eval", "summary")
N_CHECK_BEV = 8

PACKED_SPECS = {          # name -> (per-row shape, dtype, fill)
    "cand": ((K, 8, 3), np.float32, np.nan), "cand_idx": ((K,), np.int16, -1),
    "v2_final": ((K,), np.float32, np.nan), "v2_im": ((K,), np.float32, np.nan),
    "v2_sim": ((K, 5), np.float32, np.nan), "status": ((8,), np.float32, np.nan),
    "gt_traj": ((8, 3), np.float32, np.nan), "sub_traj": ((8, 3), np.float32, np.nan),
    "ok": ((), np.bool_, False),
}
INFER_SPECS = {
    "score_logit": ((K, 5), np.float16, np.nan), "z_lon": ((K, 6), np.float32, np.nan),
    "w_lat": ((K, 6), np.float32, np.nan), "c_lon": ((K, 6), np.float32, np.nan),
    "e_lat": ((K, 6), np.float32, np.nan), "corr_traj": ((K, 8, 3), np.float32, np.nan),
    "corr_score_logit": ((K, 5), np.float16, np.nan), "done": ((), np.bool_, False),
}
# eval-output key -> (array, dir) for the writer
PRED_TO_PACKED = {"ck_cand": "cand", "ck_cand_idx": "cand_idx", "ck_v2_final": "v2_final", "ck_v2_im": "v2_im",
                  "ck_v2_sim": "v2_sim"}
PRED_TO_INFER = {"ck_score_logit": "score_logit", "ck_z_lon": "z_lon", "ck_w_lat": "w_lat", "ck_c_lon": "c_lon",
                 "ck_e_lat": "e_lat", "ck_corr_traj": "corr_traj", "ck_corr_score_logit": "corr_score_logit"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T', time.gmtime())} UTC] [eval_e2e] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------- paths
class Paths:
    def __init__(self, run_dir, split: str):
        self.run_dir = Path(run_dir)
        self.name = self.run_dir.name
        self.split = split
        self.eval = self.run_dir / "eval"
        self.ck_run = self.eval / "ck_run"
        self.root = self.eval / "root"
        self.packed = self.root / "packed" / split
        self.infer = self.root / "infer" / self.name / split
        self.labels = self.root / "labels" / split
        self.out = self.root / "eval" / self.name / split
        self.hydra_cfg = self.run_dir / "train" / "code" / "hydra" / "config.yaml"
        self.pin = self.eval / "ckpt.json"


# ----------------------------------------------------------------------------------------------- ckpt
def resolve_ckpt(run_dir: Path, spec: str) -> Path:
    from tools.ck.e2e import launch_util as LU
    if spec in ("", "last"):
        p = Path(LU.find_last_ckpt(str(run_dir)))
    elif spec.startswith("epoch") or spec.isdigit():
        e = int(spec.replace("epoch", "").strip(":=_ "))
        hits = [p for ep, p in LU.epoch_ckpts(str(run_dir)) if ep == e]
        if not hits:
            raise SystemExit(f"no epoch={e} checkpoint under {run_dir}/train")
        p = hits[-1]
    else:
        p = Path(spec)
    if not p.is_file():
        raise SystemExit(f"checkpoint not found: {p}")
    return p.resolve()


def pinned_ckpt(P: Paths, spec: Optional[str]) -> Path:
    """--ckpt given -> resolve + pin; else the pinned one; else 'last' (pinned)."""
    pin = U.read_json(P.pin)
    if spec or not pin:
        p = resolve_ckpt(P.run_dir, spec or "last")
        if pin and pin.get("path") != str(p):
            log(f"ckpt changed: pinned {pin.get('path')} -> {p}")
        U.write_json(P.pin, {"path": str(p), "spec": spec or "last", "sha16": U.sha256_file(p),
                             "size": p.stat().st_size, "pinned_utc": time.strftime("%F %T", time.gmtime())})
        return p
    return Path(pin["path"])


# ----------------------------------------------------------------------------------------------- extract
def extract_ck(ckpt: Path, out_dir: Path, run_name: str = "", ck_cfg: Optional[Dict[str, Any]] = None) -> Dict:
    """Lightning ckpt -> load_ck run dir (arm S).  Returns the written config."""
    import torch
    from navsim.agents.para_ssr.ck.model import CKNet, load_ck, save_ck
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    st = {k[len(STUDENT_PREFIX):]: v for k, v in sd.items() if k.startswith(STUDENT_PREFIX)}
    if not st:
        raise SystemExit(f"{ckpt}: no '{STUDENT_PREFIX}*' keys (not a ck_e2e checkpoint)")
    ck_cfg = dict(ck_cfg or {})
    seed = int(ck_cfg.get("seed", 0) or 0)
    net = CKNet("S", seed, None, score_hidden=256, lead_aux=False)
    net.load_state_dict(st, strict=True)
    cfg = {"arm": "S", "run": run_name or out_dir.parent.parent.name, "seed": seed, "score_hidden": 256, "lead_aux": 0,
           "k": K, "e2e": True, "source_ckpt": str(ckpt), "source_ckpt_sha16": U.sha256_file(ckpt),
           "n_tensors": len(st), "ck_e2e": ck_cfg, "extracted_utc": time.strftime("%F %T", time.gmtime())}
    out_dir.mkdir(parents=True, exist_ok=True)
    save_ck(out_dir / "ckpt_last.pt", net, cfg)
    U.write_json(out_dir / "config.json", cfg)
    net2, _ = load_ck(out_dir, "last", "cpu")
    a, b = net.state_dict(), net2.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), "load_ck round trip differs"
    log(f"extracted {len(st)} CK tensors from {ckpt} -> {out_dir}")
    return cfg


def hydra_ck_cfg(P: Paths) -> Dict[str, Any]:
    if not P.hydra_cfg.is_file():
        return {}
    from omegaconf import OmegaConf
    c = OmegaConf.load(P.hydra_cfg)
    try:
        ck = c.agent.config.get("ck_e2e")
    except Exception:       # noqa: BLE001
        ck = None
    return OmegaConf.to_container(ck, resolve=False) if ck is not None else {}


# ----------------------------------------------------------------------------------------------- writer
class EvalRootWriter:
    """Row-aligned memmaps of one split in an eval root (packed + infer formats).  Rows = tokens.parquet row order.
    Several shard processes may open the same files (created under a file lock); each writes disjoint rows."""

    def __init__(self, P: Paths, tdf, ck_run: Optional[Path] = None):
        import pandas as pd
        self.P = P
        n = len(tdf)
        self.n = n
        P.packed.mkdir(parents=True, exist_ok=True)
        P.infer.mkdir(parents=True, exist_ok=True)
        tdf = tdf[["token", "log", "city"]].reset_index(drop=True).copy()
        tdf["row"] = np.arange(n, dtype=np.int64)
        with U.FileLock(P.packed / ".lock"):
            for d in (P.packed, P.infer):
                tp = d / "tokens.parquet"
                if tp.is_file():
                    old = pd.read_parquet(tp)
                    if len(old) != n or not (old.token.astype(str).to_numpy() == tdf.token.to_numpy()).all():
                        raise SystemExit(f"{tp} holds a different token list (n {len(old)} vs {n}); use --fresh")
                else:
                    tmp = d / f".tokens.tmp{os.getpid()}.parquet"
                    tdf.to_parquet(tmp, index=False)
                    os.replace(tmp, tp)
            self.A = {k: U.open_memmap(P.packed / f"{k}.npy", (n,) + s, d, fill=f) for k, (s, d, f) in
                      PACKED_SPECS.items()}
            self.I = {k: U.open_memmap(P.infer / f"{k}.npy", (n,) + s, d, fill=f) for k, (s, d, f) in
                      INFER_SPECS.items()}
        self.ck_run = ck_run

    def write(self, rows: np.ndarray, pred: Dict[str, np.ndarray], status: np.ndarray, gt: np.ndarray,
              traj: np.ndarray) -> np.ndarray:
        """rows int [B]; pred: ck_* arrays [B, ...]; status [B, 8]; gt [B, 8, 3]; traj [B, 8, 3] -> ok [B]."""
        rows = np.asarray(rows, np.int64)
        ok = np.isfinite(pred["ck_cand"]).reshape(len(rows), -1).all(1)
        ok &= np.isfinite(pred["ck_score_logit"]).reshape(len(rows), -1).all(1)
        for src, dst in PRED_TO_PACKED.items():
            self.A[dst][rows] = np.asarray(pred[src]).astype(self.A[dst].dtype)
        self.A["status"][rows] = status
        self.A["gt_traj"][rows] = gt
        self.A["sub_traj"][rows] = traj
        for src, dst in PRED_TO_INFER.items():
            self.I[dst][rows] = np.asarray(pred[src]).astype(self.I[dst].dtype)
        self.A["ok"][rows] = ok
        self.I["done"][rows] = ok
        return ok

    def flush(self) -> None:
        for m in list(self.A.values()) + list(self.I.values()):
            m.flush()

    def done_rows(self) -> np.ndarray:
        return np.asarray(self.I["done"], bool)

    def finalize_meta(self, extra: Dict[str, Any]) -> None:
        self.flush()
        ok = np.asarray(self.A["ok"], bool)
        U.write_json(self.P.packed / "meta.json", dict(
            split=self.P.split, n=self.n, k=K, n_ok=int(ok.sum()), source="eval_e2e in-process dump",
            v2_ckpt_sha16=extra.get("ckpt_sha16"), created=U.now(), **extra))
        np.save(self.P.packed / "ok_rows.npy", np.flatnonzero(ok).astype(np.int64))
        meta = U.read_json(self.P.infer / "meta.json", {}) or {}
        meta.update(run=str(self.ck_run or ""), which="last", arm="S", split=self.P.split, n=self.n, k=K, k2=0,
                    slope=0.0, rows=None, source="eval_e2e in-process (CKE2E.infer, fp32)",
                    ckpt_sha256=U.sha256_file(Path(self.ck_run) / "ckpt_last.pt", 64)
                    if self.ck_run and (Path(self.ck_run) / "ckpt_last.pt").is_file() else None, **extra)
        meta.setdefault("passes", {})["score_logit"] = dict(traj="packed cand", extra=None, decode=True,
                                                            rescore_corr=True, done="done")
        U.write_json(self.P.infer / "meta.json", meta)


# ----------------------------------------------------------------------------------------------- dump
def split_table(split: str, limit: int):
    from tools.ck.data import common as CM
    tdf = CM.split_tokens(split)
    if limit:
        tdf = tdf.iloc[:limit].reset_index(drop=True)
    return tdf


def shard_positions(tdf, shard: int, nshard: int) -> np.ndarray:
    from tools.ck.data import dump_v2 as DV
    toks = set(DV.shard_tokens(tdf, shard, nshard))
    return np.array([i for i, t in enumerate(tdf.token) if t in toks], np.int64)


def check_cand(pr: Dict[str, Any], anchors=None) -> float:
    """ck_cand == gather(anchors + offset, topk16(final)) and ck_cand[:, 0] == trajectory.  Returns top-1 error."""
    import torch
    final = pr["plan_final_rewards"].float()
    top = final.topk(K, dim=-1).indices
    if anchors is None and "trajectory_anchors" in pr:
        anchors = pr["trajectory_anchors"]
    anchors = anchors.float().to(final.device) if anchors is not None else None
    cand = pr["ck_cand"].float()
    if anchors is not None:
        ref = (anchors.reshape(1, -1, 8, 3) + pr["trajectory_offset"].float()).gather(
            1, top[:, :, None, None].expand(-1, -1, 8, 3))
        e = float((ref - cand).abs().max())
        assert e < 1e-5, f"ck_cand != gather(anchors + offset, topk) ({e})"
    assert torch.equal(pr["ck_cand_idx"].long(), top), "ck_cand_idx != topk(plan_final_rewards)"
    err1 = float((cand[:, 0] - pr["trajectory"].float()).abs().max())
    assert err1 < 1e-4, f"top-1 candidate != submitted trajectory ({err1})"
    return err1


def run_dump_shard(P: Paths, ckpt: Path, tdf, shard: int, nshard: int, batch_size: int, workers: int) -> Dict:
    import torch
    from dataclasses import replace
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.script import run_aux_evaluation as RAE
    from tools.ck.data import common as CM
    from tools.ck.data import dump_v2 as DV

    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis.strip() == "":
        dev = torch.device("cpu")
        log("no CUDA_VISIBLE_DEVICES: CPU dump (tests / tiny smoke only)")
    else:
        U.gpu_guard("cuda")
        dev = torch.device("cuda:0")
    W = EvalRootWriter(P, tdf, P.ck_run)
    pos = shard_positions(tdf, shard, nshard)
    pos = pos[~W.done_rows()[pos]]
    log(f"dump {P.split} shard {shard}/{nshard}: todo {len(pos)} of {len(tdf)} on {dev} (ckpt {ckpt})")
    if len(pos) == 0:
        return {"shard": shard, "n_done": 0}
    t0 = time.time()
    agent = RAE._build_agent(P.hydra_cfg, ckpt, dev)
    if getattr(agent, "ck_student", None) is None:
        raise SystemExit("agent has no ck_student: training config without ck_e2e.enabled")
    sp = CM.split(P.split)
    todo_toks = [str(t) for t in tdf.token.to_numpy()[pos]]
    sf = instantiate(OmegaConf.load(CM.resolve_path(sp["scene_filter"])))
    need = set(tdf.log.to_numpy()[pos])
    sf = replace(sf, tokens=list(todo_toks), max_scenes=None)
    sf = replace(sf, log_names=sorted(set(sf.log_names or need) & need))
    loader = SceneLoader(data_path=Path(sp["navsim_logs"]), sensor_blobs_path=Path(sp["sensor_blobs"]),
                         scene_filter=sf, sensor_config=agent.get_sensor_config())
    miss = set(todo_toks) - set(loader.tokens)
    assert not miss, f"scene loader misses {len(miss)} tokens e.g. {sorted(miss)[:3]}"
    log(f"agent + scene loader ready in {time.time() - t0:.0f}s")
    try:
        anchors = agent.para_ssr_model.pts_bbox_head.anchor_planner.trajectory_anchors.detach().float().to(dev)
    except AttributeError:
        anchors = None
    row_of = {t: int(p) for t, p in zip(todo_toks, pos)}
    ds = DV.FeatDS(loader, agent.get_feature_builders(), todo_toks)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=workers, collate_fn=DV.collate,
                                     pin_memory=dev.type == "cuda", prefetch_factor=2 if workers else None)
    chk_path = P.infer / "check_bev.npz"
    chk_bev, chk_rows = [], []
    n, n_bad, max_top1, t1 = 0, 0, 0.0, time.time()
    for btok, feats, gt in dl:
        fg = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else v.dtype, non_blocking=True)
              for k, v in feats.items()}
        with torch.inference_mode():
            pr = agent(fg)
            if "ck_cand" not in pr:         # infer_outputs off in the stored config: call the hook directly
                pr.update(agent._ck_e2e.infer(agent.ck_student, fg, pr))
        max_top1 = max(max_top1, check_cand(pr, anchors))
        rows = np.array([row_of[t] for t in btok], np.int64)
        npd = {k: pr[k].detach().float().cpu().numpy() if pr[k].is_floating_point() else pr[k].cpu().numpy()
               for k in list(PRED_TO_PACKED) + list(PRED_TO_INFER)}
        ok = W.write(rows, npd, fg["status_feature"].float().cpu().numpy(), gt.numpy().astype(np.float32),
                     pr["trajectory"].float().cpu().numpy())
        n_bad += int((~ok).sum())
        if shard == 0 and not chk_path.is_file() and len(chk_rows) < N_CHECK_BEV:
            from navsim.agents.para_ssr.ck.online import bev_sgrid
            g = bev_sgrid(pr["bev_embed"].float()).cpu().numpy()
            take = min(N_CHECK_BEV - len(chk_rows), len(rows))
            chk_bev.append(g[:take])
            chk_rows += rows[:take].tolist()
            if len(chk_rows) >= N_CHECK_BEV:
                tmp = chk_path.with_name(f".check_bev.tmp{os.getpid()}.npz")
                with open(tmp, "wb") as f:
                    np.savez(f, rows=np.asarray(chk_rows, np.int64), bev=np.concatenate(chk_bev, 0).astype(np.float32))
                os.replace(tmp, chk_path)
        n += len(rows)
        if n % 400 < len(rows) or n == len(pos):
            W.flush()
            el = time.time() - t1
            log(f"{P.split} s{shard} {n}/{len(pos)} {n / max(el, 1e-6):.2f} tok/s top1_err {max_top1:.1e} "
                f"not-ok {n_bad}")
    W.flush()
    meta = {"shard": shard, "nshard": nshard, "n_done": n, "n_not_ok": n_bad, "max_top1_err": max_top1,
            "seconds": round(time.time() - t0, 1), "gpu": vis, "ckpt": str(ckpt), "finished": U.now()}
    U.write_json(P.packed / f"meta_shard{shard}.json", meta)
    return meta


def stage_dump(P: Paths, ckpt: Path, a) -> None:
    tdf = split_table(P.split, a.limit)
    gpus = [g for g in a.gpus.split(",") if g.strip()] if a.gpus else []
    for g in gpus:
        if g not in U.ALLOWED_GPUS:
            raise SystemExit(f"GPU {g} not allowed (0-3 only)")
    if a.shard is not None:                                  # child process
        run_dump_shard(P, ckpt, tdf, a.shard, a.nshard, a.batch_size, a.workers)
        return
    if len(gpus) <= 1:
        if gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        run_dump_shard(P, ckpt, tdf, 0, 1, a.batch_size, a.workers)
    else:
        EvalRootWriter(P, tdf, P.ck_run)                     # create files once before the children
        procs = []
        for i, g in enumerate(gpus):
            cmd = [PY, __file__, "--run-dir", str(P.run_dir), "--split", P.split, "--stage", "dump",
                   "--ckpt", str(ckpt), "--shard", str(i), "--nshard", str(len(gpus)), "--limit", str(a.limit),
                   "--batch-size", str(a.batch_size), "--workers", str(a.workers)]
            lf = open(P.eval / f"dump_{P.split}_shard{i}.log", "a")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONPATH=CK,
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            procs.append((i, subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env), lf))
            log(f"dump shard {i}/{len(gpus)} on GPU {g} pid {procs[-1][1].pid}")
        bad = []
        for i, p, lf in procs:
            rc = p.wait()
            lf.close()
            if rc != 0:
                bad.append((i, rc))
        if bad:
            raise SystemExit(f"dump shards failed {bad} (see {P.eval}/dump_{P.split}_shard*.log)")
    W = EvalRootWriter(P, tdf, P.ck_run)
    n_done = int(W.done_rows().sum())
    pin = U.read_json(P.pin, {}) or {}
    W.finalize_meta({"ckpt": str(ckpt), "ckpt_sha16": pin.get("sha16"), "limit": a.limit, "n_done": n_done})
    log(f"dump {P.split}: {n_done}/{len(tdf)} rows done -> {P.root}")
    if n_done < len(tdf):
        raise SystemExit(f"dump incomplete: {n_done}/{len(tdf)}")


# ----------------------------------------------------------------------------------------------- label
def stage_label(P: Paths, a) -> None:
    if a.workers_label > 48:
        raise SystemExit("--workers-label <= 48")
    rows = P.packed / "ok_rows.npy"
    if not rows.is_file():
        raise SystemExit(f"{rows} missing: run --stage dump first")
    env = dict(os.environ, CK_DATA_ROOT=str(P.root), PYTHONPATH=CK, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    lab = [PY, f"{CK}/tools/ck/data/label_cands.py", "--split", P.split, "--root", str(P.root), "--workers",
           str(a.workers_label), "--rows", str(rows)]
    jobs = [("cand", lab + ["--name", "cand"]),
            (f"corr_{P.name}", lab + ["--name", f"corr_{P.name}", "--traj", str(P.infer / "corr_traj.npy")])]
    for name, cmd in jobs:
        if (P.labels / name / "labels.npy").is_file() and (P.labels / name / "meta.json").is_file():
            log(f"labels {P.split}/{name} present: skipped")
            continue
        log(f"label {P.split}/{name}: {' '.join(cmd[2:])}")
        with open(P.eval / f"label_{P.split}_{name}.log", "a") as lf:
            rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc != 0 or not (P.labels / name / "labels.npy").is_file():
            raise SystemExit(f"label_cands {name} failed rc {rc} (see {P.eval}/label_{P.split}_{name}.log)")


# ----------------------------------------------------------------------------------------------- infer_check
def stage_infer_check(P: Paths, a, tol_logit: float = 3e-2, tol_traj: float = 2e-3) -> Dict:
    import torch
    from navsim.agents.para_ssr.ck.model import load_ck
    f = P.infer / "check_bev.npz"
    if not f.is_file():
        raise SystemExit(f"{f} missing (dump shard 0 stores it)")
    z = np.load(f)
    rows, bev = z["rows"], z["bev"]
    gpus = [g for g in (a.gpus or "").split(",") if g.strip()]
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        U.gpu_guard("cuda")
        dev = torch.device("cuda:0")
    else:
        dev = torch.device("cpu")
    net, cfg = load_ck(P.ck_run, "last", str(dev))
    cand = torch.from_numpy(np.load(P.packed / "cand.npy", mmap_mode="r")[rows].astype(np.float32)).to(dev)
    status = torch.from_numpy(np.load(P.packed / "status.npy", mmap_mode="r")[rows].astype(np.float32)).to(dev)
    with torch.no_grad():
        o = net(torch.from_numpy(bev).to(dev), cand, status, decode=True, slope=0.0, rescore_corr=True)
    got = {"score_logit": o["score_logit"], "z_lon": o["z_lon"], "w_lat": o["w_lat"],
           "c_lon": o["corr"]["c_lon"][..., 2:], "e_lat": o["corr"]["e_lat"][..., 2:], "corr_traj": o["corr"]["traj"],
           "corr_score_logit": o["corr_score_logit"]}
    err = {}
    for k, v in got.items():
        ref = np.load(P.infer / f"{k}.npy", mmap_mode="r")[rows].astype(np.float32)
        err[k] = float(np.nanmax(np.abs(v.float().cpu().numpy() - ref)))
    tol = {k: (tol_logit if "logit" in k else tol_traj) for k in err}
    res = {"rows": rows.tolist(), "max_abs_err": err, "tol": tol, "device": str(dev),
           "pass": all(err[k] <= tol[k] for k in err), "ck_run": str(P.ck_run), "checked": U.now()}
    U.write_json(P.infer / "infer_check.json", res)
    log(f"infer_check {P.split}: pass={res['pass']} {err}")
    if not res["pass"]:
        raise SystemExit("infer_check failed: extracted CK run does not reproduce the in-process outputs")
    return res


# ----------------------------------------------------------------------------------------------- eval
def ensure_lead(root: Path) -> None:
    lead = root / "lead"
    if not lead.exists():
        root.mkdir(parents=True, exist_ok=True)
        os.symlink(CK_DATA / "lead", lead)


VARIANTS = ("a", "b", "c")


def select_variant(m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """navtrain_val metrics -> {variant, beta, pdms}: of a/b/c, the highest mean PDMS at its own val-best beta; ties go
    to the earlier of a, b, c (a = no correction).  None if no variant row."""
    bb = m.get("best_beta") or {}
    best = None
    for v in VARIANTS:
        if v not in bb:
            continue
        r = _row(m.get("rows") or [], v, float(bb[v]))
        if r is None or r.get("pdms") is None or not np.isfinite(r["pdms"]):
            continue
        if best is None or float(r["pdms"]) > best["pdms"]:
            best = {"variant": v, "beta": float(bb[v]), "pdms": float(r["pdms"]),
                    "rule": "max PDMS over a/b/c at each val-best beta on navtrain_val; ties -> a, b, c"}
    return best


def _record_variant(P: Paths, metrics: Path, fix: Optional[str]) -> None:
    m = json.loads(metrics.read_text())
    if P.split == "navtest":
        src = json.loads(Path(fix).read_text()) if fix and Path(fix).is_file() else {}
        m["best_variant_from_val"] = src.get("best_variant") or select_variant(src)
    else:
        m["best_variant"] = select_variant(m)
    U.write_json(metrics, m)
    log(f"variant {P.split}: {m.get('best_variant') or m.get('best_variant_from_val')}")


def stage_eval(P: Paths, a) -> Path:
    ensure_lead(P.root)
    fix = a.fix_from
    if P.split == "navtest" and not fix:
        cand = P.root / "eval" / P.name / "navtrain_val" / "metrics.json"
        if not cand.is_file():
            raise SystemExit("navtest needs --fix-from (beta / variants are chosen on navtrain_val only); run the "
                             "navtrain_val eval first")
        fix = str(cand)
    if P.split != "navtest" and fix:
        log(f"--fix-from on {P.split}: fixed betas from {fix}")
    cmd = [PY, f"{CK}/tools/ck/eval_ck.py", "--run", P.name, "--split", P.split, "--out", str(P.out),
           "--bootstrap", str(a.bootstrap)]
    if fix:
        cmd += ["--fix-from", str(fix)]
    env = dict(os.environ, CK_DATA_ROOT=str(P.root), PYTHONPATH=CK, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log(f"eval_ck {P.split}: {' '.join(cmd[2:])}")
    with open(P.eval / f"eval_{P.split}.log", "a") as lf:
        rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not (P.out / "metrics.json").is_file():
        raise SystemExit(f"eval_ck failed rc {rc} (see {P.eval}/eval_{P.split}.log)")
    _record_variant(P, P.out / "metrics.json", fix)
    return P.out / "metrics.json"


# ----------------------------------------------------------------------------------------------- summary
def _row(rows: List[Dict], variant: str, beta=None) -> Optional[Dict]:
    for r in rows:
        if r["variant"] == variant and (beta is None or (r["beta"] is not None and abs(r["beta"] - beta) < 1e-9)):
            return r
    return None


def r34_reference(split: str, per_token, n_boot: int = 2000) -> Optional[Dict]:
    """v2 r34 cand-0 official labels (Phase 1, CK_DATA) on the same tokens vs the e2e v2-head."""
    import pandas as pd
    from navsim.agents.para_ssr.ck import constants as Cn
    from tools.ck.eval_ck import log_bootstrap
    pk = CK_DATA / "packed" / split / "tokens.parquet"
    lab = CK_DATA / "labels" / split / "cand"
    if not (pk.is_file() and (lab / "labels.npy").is_file()):
        return None
    tdf = pd.read_parquet(pk)
    L = np.load(lab / "labels.npy", mmap_mode="r")
    okl = np.load(lab / "ok.npy", mmap_mode="r")
    pos = pd.Series(np.arange(len(tdf)), index=tdf.token.astype(str))
    v = per_token[per_token.variant == "v2"].drop_duplicates("token")
    idx = pos.reindex(v.token.astype(str)).to_numpy()
    m = np.isfinite(idx)
    idx = idx[m].astype(np.int64)
    v = v[m]
    keep = np.asarray(okl[idx, 0], bool)
    idx, v = idx[keep], v[keep]
    if len(idx) == 0:
        return None
    r34 = np.asarray(L[idx, 0], np.float64)
    P_ = Cn.LBL["pdms"]
    out = {"n": int(len(idx)), "source": str(lab), "note": "v2 r34(GPU 2장, 다른 seed, CK 없음): 혼재 요인 있음, 서술용"}
    for c in ("nc", "dac", "ep", "ttc", "comfort", "pdms"):
        out[f"r34_{c}"] = float(np.mean(r34[:, Cn.LBL[c]]))
    out["r34_fail_nc_or_ttc"] = float(np.mean((r34[:, Cn.LBL["nc"]] < 1) | (r34[:, Cn.LBL["ttc"]] < 1)))
    out["e2e_v2_pdms_same_tokens"] = float(v["pdms"].mean())
    out["d_pdms_e2e_minus_r34"] = log_bootstrap(v["pdms"].to_numpy(np.float64) - r34[:, P_], v.log.to_numpy(), n_boot)
    return out


def stage_summary(run_dir: Path, n_boot: int = 2000) -> Dict:
    import pandas as pd
    name = Path(run_dir).name
    ev = Path(run_dir) / "eval"
    out: Dict[str, Any] = {"run": name, "created_utc": time.strftime("%F %T", time.gmtime()),
                           "ckpt": U.read_json(ev / "ckpt.json"), "splits": {}}
    best = None
    chosen = None
    for split in ("navtrain_val", "navtest"):
        mp = ev / "root" / "eval" / name / split / "metrics.json"
        if not mp.is_file():
            continue
        m = json.loads(mp.read_text())
        rows = m["rows"]
        if split == "navtrain_val":
            best = m.get("best_beta") or {}
            chosen = m.get("best_variant") or select_variant(m)
        else:
            chosen = chosen or m.get("best_variant_from_val")
        bb = m.get("best_beta") or best or {}
        pick = {"v2-head (β=0)": _row(rows, "v2")}
        rep_key = None
        if chosen and chosen.get("variant") in bb:
            v = chosen["variant"]
            rep_key = f"{v} (val β={bb[v]:g})"
            pick[rep_key] = _row(rows, v, float(bb[v]))
        pick.update({"oracle16": _row(rows, "oracle16"), "oracle32": _row(rows, "oracle32")})
        for v in VARIANTS:
            if v in bb and f"{v} (val β={bb[v]:g})" not in pick:
                pick[f"{v} (val β={bb[v]:g})"] = _row(rows, v, float(bb[v]))
            if v not in bb or abs(float(bb[v]) - 1.0) > 1e-9:
                pick[f"{v} (β=1)"] = _row(rows, v, 1.0)
        pick = {k: r for k, r in pick.items() if r is not None}
        sp = {"n_eval": m.get("n_eval"), "n_total": m.get("n_total"), "n_logs": m.get("n_logs"),
              "best_beta_from_val": bb, "variant_from_val": chosen,
              "representative": rep_key if rep_key in pick else None, "fix_from": m.get("fix_from"), "rows": pick,
              "v2_mismatch_flag_ignored": bool(m.get("v2_mismatch")) if split == "navtest" else None}
        pt = mp.parent / "per_token.parquet"
        if pt.is_file():
            try:
                sp["r34_reference"] = r34_reference(split, pd.read_parquet(pt), n_boot)
            except Exception as e:      # noqa: BLE001
                sp["r34_reference_error"] = f"{type(e).__name__}: {e}"
        out["splits"][split] = sp
    U.write_json(ev / "summary.json", out)
    (ev / "summary.md").write_text(summary_md(out))
    log(f"summary -> {ev / 'summary.md'}")
    return out


def _pct(x) -> str:
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.2f}"


def _ci(d) -> str:
    if not isinstance(d, dict) or not d.get("n"):
        return "-"
    return f"{100 * d['mean']:+.2f} [{100 * d['lo']:+.2f}, {100 * d['hi']:+.2f}]"


def summary_md(s: Dict) -> str:
    L = [f"# CK Phase 2 e2e 평가: {s['run']}", "",
         f"ckpt: {(s.get('ckpt') or {}).get('path')} (sha16 {(s.get('ckpt') or {}).get('sha16')}). 공식 채점"
         "(score_token, LQR) [실측]. β와 변형(a/b/c)은 navtrain_val에서만 골랐다(변형 = val β에서 PDMS 최대, 동률이면 "
         "a). ★ 행이 대표 결과다. 다른 변형·β=1·oracle 행은 서술용이다(navtest에서 고르지 않는다). PDMS 등은 0-100 단위.",
         ""]
    for split, sp in s["splits"].items():
        vv = sp.get("variant_from_val") or {}
        L += [f"## {split}", "",
              f"토큰 {sp['n_eval']} / {sp['n_total']}, log {sp['n_logs']}. val에서 고른 β: {sp['best_beta_from_val']}, "
              f"변형: {vv.get('variant', '-')} (val PDMS {_pct(vv.get('pdms'))})"
              + (" (navtest의 v2_mismatch 표시는 r34 CSV 비교라 e2e ckpt에 해당하지 않는다)" if split == "navtest" else ""),
              "", "| 선택 | PDMS | ΔPDMS vs v2-head [95% CI] | NC | DAC | EP | TTC | C | NC 실패 | TTC 실패 | "
              "앞차 감속 NC/TTC 실패 (n) | Δ앞차 실패 [95% CI] | 바뀐 비율 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, r in sp["rows"].items():
            lead = f"{_pct(r.get('lead_fail_nc_ttc'))} ({r.get('lead_n')})" if r.get("lead_n") else "-"
            lab = (f"★ {k} (val 선택)" if k == sp.get("representative")
                   else k if k.startswith("v2-head") else f"{k} (서술용)")
            L.append(f"| {lab} | {_pct(r['pdms'])} | {_ci(r.get('d_pdms'))} | {_pct(r['nc'])} | {_pct(r['dac'])} | "
                     f"{_pct(r['ep'])} | {_pct(r['ttc'])} | {_pct(r['comfort'])} | {_pct(r.get('fail_nc'))} | "
                     f"{_pct(r.get('fail_ttc'))} | {lead} | {_ci(r.get('d_lead_fail'))} | "
                     f"{100 * r.get('frac_changed', 0):.1f} |")
        ref = sp.get("r34_reference")
        if ref:
            L += ["", f"비교 기준 v2 r34(cand 0, 같은 토큰 {ref['n']}): PDMS {_pct(ref['r34_pdms'])}, "
                      f"NC|TTC 실패 {_pct(ref['r34_fail_nc_or_ttc'])}. e2e v2-head − r34 = "
                      f"{_ci(ref['d_pdms_e2e_minus_r34'])} (log bootstrap). {ref['note']}."]
        L.append("")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------- main
def fresh(P: Paths) -> None:
    for d in (P.packed, P.infer, P.labels, P.out):
        if d.exists():
            log(f"--fresh: removing {d}")
            shutil.rmtree(d)


def get_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", required=True, choices=["navtrain_val", "navtest"])
    ap.add_argument("--ckpt", default="", help="last (default) | epoch:<E> | <path>; pinned in <RUN>/eval/ckpt.json")
    ap.add_argument("--stage", default="all")
    ap.add_argument("--limit", type=int, default=0, help="first N split tokens (smoke)")
    ap.add_argument("--gpus", default="0", help="dump GPUs (0-3), one process each; '' = CPU")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6, help="dataloader workers per dump process")
    ap.add_argument("--workers-label", type=int, default=48)
    ap.add_argument("--fix-from", default="")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--fresh", action="store_true", help="remove this split's eval-root outputs first")
    ap.add_argument("--shard", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--nshard", type=int, default=1, help=argparse.SUPPRESS)
    return ap


def main(argv=None) -> int:
    a = get_parser().parse_args(argv)
    P = Paths(a.run_dir, a.split)
    stages = list(STAGES) if a.stage == "all" else [s.strip() for s in a.stage.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; known {STAGES}")
    if a.fresh and a.shard is None:
        fresh(P)
    P.eval.mkdir(parents=True, exist_ok=True)
    ckpt = None
    if any(s in stages for s in ("extract", "dump")):
        ckpt = Path(a.ckpt) if a.shard is not None else pinned_ckpt(P, a.ckpt or None)
    for s in stages:
        t0 = time.time()
        if s == "extract":
            extract_ck(ckpt, P.ck_run, P.name, hydra_ck_cfg(P))
        elif s == "dump":
            stage_dump(P, ckpt, a)
        elif s == "label":
            stage_label(P, a)
        elif s == "infer_check":
            stage_infer_check(P, a)
        elif s == "eval":
            stage_eval(P, a)
        elif s == "summary":
            stage_summary(P.run_dir, a.bootstrap)
        log(f"stage {s} done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
